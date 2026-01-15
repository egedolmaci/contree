import pandas as pd
import ast
import os
import sys

# --- AYARLAR ---
DATASET_DIR = "/home/cs406keremtufan/contree/datasets"
BENCHMARK_FILE = "openmp.csv"

def smart_load_dataset(dataset_name):
    filepath = os.path.join(DATASET_DIR, dataset_name)
    if not os.path.exists(filepath):
        return None, None, None, "File Not Found"

    delimiters = [',', '\s+', ';', '\t']
    
    for sep in delimiters:
        try:
            # Tüm satırları oku
            df = pd.read_csv(filepath, header=None, sep=sep, engine='python')
            
            if df.shape[1] > 1:
                # --- LABEL KONUMU TESPİTİ ---
                # Hangi sütun label? Genelde label integer ve az sayıda unique değere sahiptir.
                first_col = df.iloc[:, 0]
                last_col = df.iloc[:, -1]
                
                # Basit Hevristik: Unique değer sayısı az olan ve tercihen integer olan labeldır.
                # Eğer ikisi de benzerse varsayılan 'son'dur, ama burada 'ilk'i deneyeceğiz.
                
                is_first_integer = pd.api.types.is_integer_dtype(first_col) or first_col.nunique() < 20
                is_last_integer = pd.api.types.is_integer_dtype(last_col) or last_col.nunique() < 20
                
                # Senin veri setlerinde sample '0 0.812...' olduğu için Label başta görünüyor.
                # Önceliği 'İlk Sütun'a verelim eğer integer gibiyse.
                
                label_loc = "LAST"
                if is_first_integer and not is_last_integer:
                    label_loc = "FIRST"
                elif is_first_integer and is_last_integer:
                    # İkisi de olabilir, veri formatına bak. Senin formatında Label First yaygın.
                    # Karışıklığı önlemek için script içinde her iki durumu deneyip skoru tutanını seçeceğiz.
                    label_loc = "TRY_BOTH" 
                
                return df, label_loc, None, f"Shape {df.shape}"
        except:
            continue

    return None, None, None, "Parse Failed"

def prepare_data(df, label_position):
    """Veriyi belirtilen label konumuna göre X ve y olarak ayırır."""
    if label_position == "FIRST":
        y_raw = df.iloc[:, 0]
        X = df.iloc[:, 1:] 
        # Feature indekslerini düzeltmek gerekebilir mi? 
        # C++ kodu label'ı atıp featureları 0'dan mı başlatıyor? Genelde evet.
        # X'in sütun isimlerini 0..N olarak resetle
        X.columns = range(X.shape[1])
    else: # LAST
        y_raw = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        X.columns = range(X.shape[1])
        
    # Label Normalize (0, 1, 2...)
    y_normalized, uniques = pd.factorize(y_raw, sort=True)
    y = pd.Series(y_normalized)
    
    return X, y

def predict_and_count_errors(node, X_subset, y_subset):
    if len(y_subset) == 0: return 0
    
    if isinstance(node, (int, float)):
        return (y_subset != node).sum()

    if isinstance(node, list):
        if len(node) < 3 or (node[2] == -1 and node[3] == -1):
            majority = y_subset.mode()
            if len(majority) > 0: return (y_subset != majority[0]).sum()
            return 0
        
        feature_idx = int(node[0])
        threshold = float(node[1])
        
        if feature_idx >= X_subset.shape[1]:
            raise IndexError
            
        mask_left = X_subset.iloc[:, feature_idx] <= threshold
        
        err_l = predict_and_count_errors(node[2], X_subset[mask_left], y_subset[mask_left])
        err_r = predict_and_count_errors(node[3], X_subset[~mask_left], y_subset[~mask_left])
        return err_l + err_r
    return 0

def parse_tree(tree_str):
    try:
        if pd.isna(tree_str) or tree_str == "ERROR": return None
        return ast.literal_eval(tree_str)
    except: return None

def main():
    if not os.path.exists(BENCHMARK_FILE):
        print(f"Hata: {BENCHMARK_FILE} bulunamadı.")
        return

    results = pd.read_csv(BENCHMARK_FILE)
    
    print(f"{'DATASET':<12} | {'D':<2} | {'MODE':<3} | {'REPORT':<6} | {'CALC':<6} | {'STATUS'} | {'NOTE'}")
    print("-" * 80)

    for index, row in results.iterrows():
        dataset = row['Dataset']
        rep_score = row['Misclassification']
        tree_str = row['TreeStructure']
        
        tree = parse_tree(tree_str)
        if tree is None: continue

        df, label_loc, _, msg = smart_load_dataset(dataset)
        if df is None: continue

        # Hangi Label Konumu Doğru? Deneme Yanılma
        possible_locs = ["FIRST", "LAST"] if label_loc == "TRY_BOTH" else [label_loc]
        if label_loc == "LAST" and "0 0.8" in str(df.head(1)): possible_locs = ["FIRST", "LAST"] # Override for tricky files

        best_score = -1
        best_status = "❌"
        best_diff = 999999
        final_loc = "?"

        for loc in possible_locs:
            try:
                X, y = prepare_data(df, loc)
                calc = predict_and_count_errors(tree, X, y)
                
                diff = abs(calc - rep_score)
                if diff < best_diff:
                    best_diff = diff
                    best_score = calc
                    final_loc = loc
                    
                if diff == 0: break # Mükemmel eşleşme bulundu
            except:
                continue
        
        # Sonuç Yazdır
        status = "❌ DIFF"
        if best_diff == 0: status = "✅ MATCH"
        elif best_diff < 10: status = "⚠️ CLOSE"
        
        print(f"{dataset:<12} | {row['Depth']:<2} | {row['Mode']:<3} | {rep_score:<6} | {best_score:<6} | {status} | Loc:{final_loc}")

if __name__ == "__main__":
    main()