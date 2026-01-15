#include <benchmark/benchmark.h>
#include <omp.h>

  #include "configuration.h"
  #include "dataset.h"
  #include "dataview.h"
  #include "file_reader.h"
  #include "general_solver.h"
  #include "tree.h"

#ifdef USE_CUDA
  #include "GPUBruteForceSolver.h"
#endif

  #include <climits>
  #include <memory>
  #include <string>

  // Dataset configurations
  struct DatasetConfig {
      std::string name;
      std::string path;
  };

  // Small and medium datasets only (8 total)
  static const DatasetConfig datasets[] = {
      // Small datasets
      {"bank", "../../datasets/bank.txt"},           // 108 KB
      {"raisin", "../../datasets/raisin.txt"},       // 123 KB
      {"wilt", "../../datasets/wilt.txt"},           // 486 KB
      {"rice", "../../datasets/rice.txt"},           // 518 KB
      // Medium datasets
      {"segment", "../../datasets/segment.txt"},     // 754 KB
      {"bidding", "../../datasets/bidding.txt"},     // 861 KB
      {"fault", "../../datasets/fault.txt"},         // 938 KB
      {"page", "../../datasets/page.txt"},           // 1.1 MB
  };

  struct RunResult {
      double accuracy;
      int misclassification;
      int instance_count;
  };

  constexpr double kBenchmarkTimeLimitSeconds = 120.0;

  static Configuration MakeConfig(int max_depth, int max_gap, bool use_gpu) {
      Configuration config;
      config.max_depth = max_depth;
      config.max_gap = max_gap;
      config.use_upper_bound = true;
      config.sort_gini = false;
      config.print_logs = false;
      config.use_gpu_bruteforce = use_gpu;
      config.max_thresholds_per_feature = 128;
      config.stopwatch.Initialise(kBenchmarkTimeLimitSeconds);
      return config;
  }

  // Helper function
  static RunResult RunConTreeTraining(Dataset& unsorted_dataset, int class_number, int max_depth, int max_gap, bool use_gpu) {
      Dataset sorted_dataset = unsorted_dataset;
      sorted_dataset.sort_feature_values();

      Configuration config = MakeConfig(max_depth, max_gap, use_gpu);

      Dataview dataview = Dataview(&sorted_dataset, &unsorted_dataset, class_number, config.sort_gini);

      auto optimal_tree = std::make_shared<Tree>();
      config.is_root = true;
      GeneralSolver::create_optimal_decision_tree(dataview, config, optimal_tree, INT_MAX);

      const int instance_count = dataview.get_dataset_size();
      const int misclassification = optimal_tree->misclassification_score;
      const double accuracy = (instance_count > 0)
          ? (double(instance_count - misclassification) / double(instance_count))
          : 0.0;

      benchmark::DoNotOptimize(optimal_tree);
      return {accuracy, misclassification, instance_count};
  }

  // Parameterized benchmark
  static void BM_ConTree_CPU(benchmark::State& state) {
      static bool printed_threads = false;
      if (!printed_threads) {
          #pragma omp parallel
          {
              #pragma omp single
              std::cout << "[BENCH] OMP threads = " << omp_get_num_threads()
                        << " / procs = " << omp_get_num_procs() << "\n";
          }
          printed_threads = true;
      }
      // Extract parameters
      int dataset_idx = state.range(0);
      int depth = state.range(1);
      int max_gap = 0;  // Always 0 for optimal trees

      const auto& dataset = datasets[dataset_idx];
      Dataset unsorted_dataset;
      int class_number = -1;
      file_reader::read_file(dataset.path, unsorted_dataset, class_number);

      // Set benchmark name to be readable
      state.SetLabel(dataset.name + "_d" + std::to_string(depth) + "_cpu");

      RunResult result{0.0, 0, 0};
      for (auto _ : state) {
          result = RunConTreeTraining(unsorted_dataset, class_number, depth, max_gap, false);
      }

      state.counters["accuracy"] = result.accuracy;
      state.counters["misclass"] = result.misclassification;
      state.counters["instances"] = result.instance_count;
  }

  BENCHMARK(BM_ConTree_CPU)
      // Small datasets
      ->Args({0, 3})  // bank, depth 3
      ->Args({0, 4})  // bank, depth 4
      ->Args({1, 3})  // raisin, depth 3
      ->Args({1, 4})  // raisin, depth 4
      ->Args({2, 3})  // wilt, depth 3
      ->Args({2, 4})  // wilt, depth 4
      ->Args({3, 3})  // rice, depth 3
      ->Args({3, 4})  // rice, depth 4
      // Medium datasets
      ->Args({4, 3})  // segment, depth 3
      ->Args({4, 4})  // segment, depth 4
      ->Args({5, 3})  // bidding, depth 3
      ->Args({5, 4})  // bidding, depth 4
      ->Args({6, 3})  // fault, depth 3
      ->Args({6, 4})  // fault, depth 4
      ->Args({7, 3})  // page, depth 3
      ->Args({7, 4})  // page, depth 4
      ->Unit(benchmark::kSecond);

#ifdef USE_CUDA
  static void BM_ConTree_GPU(benchmark::State& state) {
      static bool printed_threads = false;
      if (!printed_threads) {
          #pragma omp parallel
          {
              #pragma omp single
              std::cout << "[BENCH] OMP threads = " << omp_get_num_threads()
                        << " / procs = " << omp_get_num_procs() << "\n";
          }
          printed_threads = true;
      }
      // Extract parameters
      int dataset_idx = state.range(0);
      int depth = state.range(1);
      int max_gap = 0;  // Always 0 for optimal trees

      const auto& dataset = datasets[dataset_idx];
      Dataset unsorted_dataset;
      int class_number = -1;
      file_reader::read_file(dataset.path, unsorted_dataset, class_number);

      // Set benchmark name to be readable
      state.SetLabel(dataset.name + "_d" + std::to_string(depth) + "_gpu");

      RunResult result{0.0, 0, 0};
      bool gpu_initialized = false;
      for (auto _ : state) {
          if (!gpu_initialized) {
              state.PauseTiming();
              Configuration init_config = MakeConfig(depth, max_gap, true);
              Dataset sorted_for_gpu = unsorted_dataset;
              sorted_for_gpu.sort_feature_values();
              Dataview full_view = Dataview(&sorted_for_gpu, &unsorted_dataset, class_number, init_config.sort_gini);
              GPUBruteForceSolver::Initialize(full_view, init_config);
              state.ResumeTiming();
              gpu_initialized = true;
          }
          result = RunConTreeTraining(unsorted_dataset, class_number, depth, max_gap, true);
      }

      GPUBruteForceSolver::FreeMemory();

      state.counters["accuracy"] = result.accuracy;
      state.counters["misclass"] = result.misclassification;
      state.counters["instances"] = result.instance_count;
  }

  BENCHMARK(BM_ConTree_GPU)
      // Small datasets
      ->Args({0, 3})  // bank, depth 3
      ->Args({0, 4})  // bank, depth 4
      ->Args({1, 3})  // raisin, depth 3
      ->Args({1, 4})  // raisin, depth 4
      ->Args({2, 3})  // wilt, depth 3
      ->Args({2, 4})  // wilt, depth 4
      ->Args({3, 3})  // rice, depth 3
      ->Args({3, 4})  // rice, depth 4
      // Medium datasets
      ->Args({4, 3})  // segment, depth 3
      ->Args({4, 4})  // segment, depth 4
      ->Args({5, 3})  // bidding, depth 3
      ->Args({5, 4})  // bidding, depth 4
      ->Args({6, 3})  // fault, depth 3
      ->Args({6, 4})  // fault, depth 4
      ->Args({7, 3})  // page, depth 3
      ->Args({7, 4})  // page, depth 4
      ->Unit(benchmark::kSecond);
#endif

  BENCHMARK_MAIN();
