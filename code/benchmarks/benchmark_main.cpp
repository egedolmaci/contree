  #include <benchmark/benchmark.h>

  #include "configuration.h"
  #include "dataset.h"
  #include "dataview.h"
  #include "file_reader.h"
  #include "general_solver.h"
  #include "tree.h"

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

  // Helper function
  static void RunConTreeTraining(const std::string& dataset_path, int max_depth, int max_gap) {
      Dataset unsorted_dataset;
      int class_number = -1;
      file_reader::read_file(dataset_path, unsorted_dataset, class_number);

      Dataset sorted_dataset = unsorted_dataset;
      sorted_dataset.sort_feature_values();

      Configuration config;
      config.max_depth = max_depth;
      config.max_gap = max_gap;
      config.use_upper_bound = true;
      config.sort_gini = false;
      config.print_logs = false;
      config.stopwatch.Initialise(120.0);  // 2 minute timeout per benchmark

      Dataview dataview = Dataview(&sorted_dataset, &unsorted_dataset, class_number, config.sort_gini);

      auto optimal_tree = std::make_shared<Tree>();
      config.is_root = true;
      GeneralSolver::create_optimal_decision_tree(dataview, config, optimal_tree, INT_MAX);

      benchmark::DoNotOptimize(optimal_tree);
  }

  // Parameterized benchmark
  static void BM_ConTree(benchmark::State& state) {
      // Extract parameters
      int dataset_idx = state.range(0);
      int depth = state.range(1);
      int max_gap = 0;  // Always 0 for optimal trees

      const auto& dataset = datasets[dataset_idx];

      // Set benchmark name to be readable
      state.SetLabel(dataset.name + "_d" + std::to_string(depth));

      for (auto _ : state) {
          RunConTreeTraining(dataset.path, depth, max_gap);
      }
  }

  BENCHMARK(BM_ConTree)
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

  BENCHMARK_MAIN();
