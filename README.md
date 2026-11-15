# SINYAL · EarlySkill-ID — Deteksi Dini Emerging IT Skills Menggunakan Analisis Temporal Multi-Sinyal dan Machine Learning pada Job Postings

**Peneliti:** Bintang Maulana (`NPM: 2313020011`)

**Program Studi:** Teknik Informatika (Kelas 4A) — Fakultas Teknik dan Ilmu Komputer, Universitas Nusantara PGRI Kediri

**Metode Penelitian:** *Research and Development (R&D — Engineering Approach)* dengan *Prototype Model* (Pressman & Maxim, 2020) sesuai Pedoman KTI 2026

**Versi Prototipe & Pipeline:** `v2026.4-DualCorpus` (4 Fitur Inti Multi-Indikator · Global Benchmark 2021–2026 & Filter Regional Indonesia 2023–2026)

## 1. Gambaran Umum Penelitian & Batasan Sistem

Repositori ini memuat kode sumber lengkap **Pipeline Eksperimen Jupyter Notebook (`.ipynb`)** serta **Aplikasi Web Prototipe `SINYAL · EarlySkill-ID`** untuk mendeteksi secara dini keterampilan Teknologi Informasi (*emerging IT skills*) yang sedang tumbuh sebelum mencapai saturasi pasar umum dalam horizon prediksi **H = 2 kuartal ke depan**.

Sistem dirancang secara ketat sebagai **Sistem Peringatan Dini (*Early-Warning Prediction System*)** bagi mahasiswa Teknik Informatika dan *fresh graduate* TI untuk mengamati sinyal pergerakan teknologi baru berbasis data lowongan kerja aktual, **bukan** sistem rekomendasi karier preskriptif atau penyusun jalur pembelajaran (*learning path*).

Keunggulan utama arsitektur ini meliputi:

* **Saringan Kandidat *Risk Set* Berbasis Kuantil Kausal Pra-Evaluasi (Adaptasi Yazdanian et al., 2022):** Memisahkan keterampilan yang masih berada pada fase kemunculan awal (`causal_roll4_share <= P60` dan `causal_cummax_share <= P65` pada jendela jangkar pra-evaluasi) dari keterampilan TI mapan seperti Python, SQL, atau JavaScript.

* **Rekayasa 4 Fitur Inti Multi-Indikator Bebas Kebocoran Masa Depan (*Strictly Causal*** $\mathcal{F}_t$**):** Menggabungkan sinyal dinamika permintaan, evolusi jaringan ko-okurensi teknologi, dan difusi lintas peran pekerjaan TI.

* **Validasi *Expanding-Window Walk-Forward* dengan *Purging Gap*** $H = 2$ **Kuartal:** Memotong 2 kuartal terakhir pada setiap jendela pelatihan agar label masa depan $y_{s,t}^{(2)}$ tidak bocor ke proses pelatihan model maupun kalibrasi ambang batas $\tau_{\text{OOF}}$.

* **Arsitektur *Dual-Corpus* (Global Benchmark & Filter Regional Indonesia):** Mendukung analisis pada korpus global sebagai *leading indicator* difusi teknologi sekaligus korpus pasar kerja TI domestik Indonesia (`2023-Q1` s.d. `2026-Q4`, dengan penyaringan otomatis yang membuang data tahun 2022 yang tidak merata).

## 2. Spesifikasi Dataset Dual-Corpus (Global & Regional Indonesia)

Pipeline notebook dan aplikasi web mengintegrasikan dua cakupan dataset yang dapat dipilih secara langsung melalui parameter konfigurasi maupun tombol **Mode Filter Dataset** di antarmuka web:

| Parameter & Karakteristik | Korpus Global Benchmark (`GLOBAL`) | Korpus Filter Regional Indonesia (`INDONESIA`) | 
| ----- | ----- | ----- | 
| **Berkas Dataset Sumber** | `theirstack_pure_combined_2021_2026.csv` | `indonesia_it_jobs_2023_2026.csv` (disaring dari `2022_2026.csv`) | 
| **Perlakuan Tahun 2022** | Dipertahankan (`2021-Q1` s.d. `2026-Q3` utuh) | **Dihapus ketat (`year >= 2023`)** — 14 baris tahun 2022 dibuang | 
| **Rentang Periode Kuartal** | `2021-Q1` s.d. `2026-Q3` (**23 kuartal** tanpa jeda) | `2023-Q1` s.d. `2026-Q4` (**16 kuartal** tanpa jeda) | 
| **Volume Lowongan Mentah** | `9.698` baris lowongan kerja TI | `6.075` baris (2023–2026) setelah membuang 14 baris 2022 | 
| **Volume Lowongan TI Valid** | **`9.690` lowongan TI valid** | **`5.926` lowongan TI valid** (141 *repost* & 8 non-TI dihapus) | 
| **Distribusi Tahunan Valid** | 2021–2026 (\~350–490 lowongan/kuartal) | 2023: `783` · 2024: `1.602` · 2025: `1.536` · 2026: `2.005` | 
| **Dimensi Panel Temporal** | `113 skill × 23 kuartal = 2.599` baris panel | `113 skill × 16 kuartal = 1.808` baris panel (`33.286` *job-skill pairs*) | 
| **Jendela Jangkar Kuantil** | `q_idx 2..6` (`2021-Q3` s.d. `2022-Q3`, *Train Fold 1*) | `q_idx 2..4` (`2023-Q3` s.d. `2024-Q1`, *Train Fold 1*) | 
| **Ambang *Risk Set* (`b_max`, `c_max`)** | `b_max = 4,74%` (`P60`) · `c_max = 7,94%` (`P65`) | `b_max = 4,70%` (`P60`) · `c_max = 8,17%` (`P65` — Pasca-Harmonisasi) | 
| **Kandidat *Risk Set* Aktif** | **73 skill kandidat** (`1.392` obs total, `1.270` berlabel) | **75 skill kandidat** (`971` obs total, `821` berlabel pasca-harmonisasi `n_skills >= 1`) | 
| **Skema Evaluasi Walk-Forward** | **6 Fold** (*Purge* $H=2$, uji `2023-Q2` s.d. `2026-Q1`) | **4 Fold** (*Purge* $H=2$, uji `2024-Q3` s.d. `2026-Q2`) | 
| **Kuartal Peramalan Prospektif** | `2026-Q2` & `2026-Q3` (target $\to$ `2026-Q4` & `2027-Q1`) | `2026-Q3` & `2026-Q4` (target $\to$ `2027-Q1` & `2027-Q2`) | 

## 3. Struktur Direktori & Berkas Proyek

Berikut adalah susunan lengkap berkas dalam repositori penelitian beserta keterkaitan antara dataset, ke-6 Jupyter Notebook, hasil ekspor eksperimen, dan aplikasi web **SINYAL · EarlySkill-ID**:

```
sinyal-earlyskill-id/
├── README.md                                                 # Dokumentasi utama proyek, notebook, & aplikasi web
│
├── data/
│   ├── raw/
│   │   ├── theirstack_pure_combined_2021_2026.csv            # Dataset Global Benchmark (9.698 baris mentah -> 9.690 valid)
│   │   ├── indonesia_it_jobs_2022_2026.csv                   # Dataset mentah Indonesia sebelum penyaringan tahun 2022 (6.089 baris)
│   │   └── indonesia_it_jobs_2023_2026.csv                   # Dataset bersih Indonesia tanpa tahun 2022 (6.075 baris -> 5.926 valid)
│   └── processed/
│       ├── skill_taxonomy_and_aliases.csv                    # Master taksonomi 113 skill kanonis lintas 8 domain TI & regex alias
│       ├── job_postings_preprocessed.csv                     # Lowongan TI valid pasca-deduplikasi & normalisasi 13 kategori peran
│       └── job_skill_provenance_long.csv                     # Tabel relasi job_id x canonical_skill x canonical_role
│
├── notebooks/                                                # Pipeline Eksperimen Jupyter Notebook (1 Utama + 5 Modular)
│   ├── Emerging_IT_Skills_Full_Pipeline_4Fitur.ipynb         # [UTAMA] Pipeline terpadu Tahap 1-9 + Mode Indonesia (8B) + 37 Unit Tests
│   ├── 01_Preprocessing_dan_Taksonomi_Skill.ipynb            # [MODUL 1] Tahap 1-3: Konfigurasi, Taksonomi 113 Skill, Normalisasi Peran
│   ├── 02_Panel_Temporal_dan_Rekayasa_Fitur.ipynb            # [MODUL 2] Tahap 4: Panel Temporal, Empirical Bayes, & 4 Fitur Inti
│   ├── 03_Pelabelan_Target_dan_Skema_Walk_Forward.ipynb      # [MODUL 3] Tahap 5: Aturan Kuantil P60/P65, Label H=2, & 6/4 Fold Purge
│   ├── 04_Evaluasi_Model_ML_dan_Studi_Ablasi.ipynb           # [MODUL 4] Tahap 6: Ablasi M1-M5 (LR, RF, XGB, LGBM) & Uji Newey-West HAC
│   └── 05_Deteksi_Dini_SHAP_dan_Prediksi_2026.ipynb          # [MODUL 5] Tahap 7-9: Lead Time, SHAP, Prediksi 2026, & Ekspor Web App
│
├── outputs/                                                  # Artefak Keluaran Eksperimen Notebook
│   ├── demo_state/                                           # Penyimpanan state berantai antar-notebook modular saat demo sidang
│   │   ├── stage01_state.pkl                                 # Output Notebook 01 -> Input Notebook 02
│   │   ├── stage02_state.pkl                                 # Output Notebook 02 -> Input Notebook 03
│   │   ├── stage03_state.pkl                                 # Output Notebook 03 -> Input Notebook 04
│   │   └── stage04_state.pkl                                 # Output Notebook 04 -> Input Notebook 05
│   ├── panel/
│   │   ├── 04_cooccurrence_network_edges_by_quarter.csv      # Sisi graf ko-okurensi Jaccard kuartalan (J_t & c_uv)
│   │   └── 05_full_dataset_features_and_labels_with_reasons.csv # Panel lengkap fitur, status in_risk_set, & target_emerging
│   ├── metrics/
│   │   ├── 01_metrics_by_model_algorithm_and_fold.csv        # Metrik PR-AUC, ROC-AUC, F1, P@5, P@10 per fold evaluasi
│   │   ├── 02_ablation_summary_m1_to_m5.csv                  # Ringkasan perbandingan konfigurasi M1 s.d. M5
│   │   ├── 03_statistical_hypothesis_tests_m1_to_m5.csv      # Hasil uji t Newey-West HAC (L=2) & Moving-Block Bootstrap
│   │   ├── 06_shap_global_feature_importance.csv             # Bobot atribusi SHAP global ke-4 fitur inti
│   │   ├── 07_shap_signal_group_importance.csv               # Bobot atribusi SHAP per kelompok sinyal (Demand, Context, Role)
│   │   └── 08_shap_local_top_predictions_explanations.csv    # Dekomposisi SHAP lokal untuk kandidat peringkat teratas
│   ├── predictions/
│   │   └── 01_oos_predictions_all_models_and_folds.csv       # Probabilitas prediksi Out-of-Sample & Prospektif beserta tau_OOF
│   ├── lead_time/
│   │   ├── 01_lead_time_per_skill_selected_model_1q_confirm.csv # Rincian t_warn, t_emerge, & lead time per skill
│   │   └── 03_lead_time_summary_comparison.csv               # Ringkasan Early Warning Detection Rate & Mean Lead Time
│   └── figures/                                              # Grafik resolusi tinggi (Gambar 1 s.d. Gambar 8 untuk Bab IV Skripsi)
│       ├── fig1_dataset_audit_and_skill_cleaning.png
│       ├── fig2_emerging_skills_trajectories.png
│       ├── fig2b_target_and_walk_forward_schedule.png
│       ├── Ablasi_M1_sd_M5_Risk_Set.png
│       └── SHAP_10_Fitur_dan_4_Sinyal_Risk_Set.png
│
├── SINYAL_EarlySkill_ID_Web_App.html                         # Aplikasi Web Interaktif Mandiri (Single-File Canvas / Browser)
└── SINYAL_EarlySkill_ID_Fullstack/                           # Direktori Paket Aplikasi Web Fullstack (di dalam .zip)
    ├── app.py                                                # Server REST API Python (melayani UI & prediksi real-time)
    ├── README.md                                             # Panduan instalasi & endpoint API lokal
    ├── trained_models_4fitur.pkl                             # Model terlatih 4 fitur inti + tau_OOF (Korpus Global)
    ├── trained_models_indo_4fitur.pkl                        # Model terlatih 4 fitur inti + tau_OOF (Korpus Indonesia 2023-2026)
    ├── earlyskill_id.db                                      # Basis data SQLite3 6 tabel PDM (Korpus Global)
    ├── earlyskill_id_indo.db                                 # Basis data SQLite3 6 tabel PDM (Korpus Indonesia 2023-2026)
    ├── webapp_bundle_4fitur.json                             # Bundel JSON lengkap antarmuka web (Korpus Global)
    ├── webapp_bundle_indo_4fitur.json                        # Bundel JSON lengkap antarmuka web (Korpus Indonesia 2023-2026)
    └── static/
        └── index.html                                        # Antarmuka web 4 halaman dengan Switcher Indonesia vs Global

```

## 4. Rincian & Panduan Eksekusi Jupyter Notebook (`.ipynb`)

Proyek ini menyediakan **1 Notebook Utama Terpadu (*Full Pipeline*)** serta **5 Notebook Modular Berurutan (`01` s.d. `05`)** yang saling terhubung melalui penyimpanan *state* otomatis (`outputs/demo_state/stage01_state.pkl` s.d. `stage04_state.pkl`) untuk memudahkan demonstrasi saat sidang skripsi:

| Nama Berkas Notebook | Cakupan Tahapan & Modul | Deskripsi Fungsional & Keluaran Utama | 
| ----- | ----- | ----- | 
| **`Emerging_IT_Skills_Full_Pipeline_4Fitur.ipynb`** | **Tahap 1 s.d. Tahap 9 + Unit Tests (Terpadu)** | Notebook lengkap ujung-ke-ujung yang memuat seluruh kode konfigurasi, ekstraksi taksonomi 113 skill, pembentukan panel, rekayasa 4 fitur inti, pelabelan kuantil kausal, studi ablasi `M1`–`M5`, analisis *Lead Time* & SHAP, **Tahap 8B (Mode Filter Regional Indonesia 2023–2026 tanpa 2022)**, **Tahap 9 (Ekspor Dual-Corpus `.pkl`, `.db`, `.json`)**, serta **37 Unit Tests**. | 
| **`01_Preprocessing_dan_Taksonomi_Skill.ipynb`** | **Tahap 1, Tahap 2, & Tahap 3** | Inisialisasi `PipelineConfig`, definisi taksonomi kanonis **113 skill TI pada 8 domain**, ekstraktor regex presisi (memisahkan entitas rancu seperti SQL vs PostgreSQL/MySQL dan Git vs GitHub/GitLab), normalisasi **13 kategori peran TI** (termasuk `Other / Unclassified IT Role`), audit *EoQ Snapshot*, dukungan pemuatan `indonesia_it_jobs_2023_2026.csv`, dan visualisasi **Gambar 1**. Menyimpan `stage01_state.pkl`. | 
| **`02_Panel_Temporal_dan_Rekayasa_Fitur.ipynb`** | **Tahap 4** | Memuat `stage01_state.pkl`, membangun panel seimbang `skill × kuartal`, menerapkan stabilisasi *Empirical Bayes Shrinkage* serta sensitivitas *company-cap* ($\le 5$ lowongan/perusahaan/kuartal), mengekstraksi **4 Fitur Inti Multi-Indikator**, mengaudit jaringan ko-okurensi Jaccard, dan memvisualisasikan trajektori pertumbuhan skill pada **Gambar 2**. Menyimpan `stage02_state.pkl`. | 
| **`03_Pelabelan_Target_dan_Skema_Walk_Forward.ipynb`** | **Tahap 5, Tahap 5B, & Tahap 5C** | Memuat `stage02_state.pkl`, menurunkan ambang *Risk Set* kausal (`P60 roll4` & `P65 cummax`) dari jendela jangkar Fold 1 tanpa kebocoran masa depan, melabeli target kemunculan $y_{s,t}^{(2)}$ pada horizon $H=2$ kuartal, memverifikasi bebas *data leakage* pada skema *Expanding Walk-Forward* (**Gambar 2B**), serta menguji ketahanan target dengan *Anti-Mean-Reversion* dan *Binomial SE Guard*. Menyimpan `stage03_state.pkl`. | 
| **`04_Evaluasi_Model_ML_dan_Studi_Ablasi.ipynb`** | **Tahap 6** | Memuat `stage03_state.pkl`, melatih dan mengevaluasi 4 keluarga algoritma (`LogisticRegression`, `RandomForest`, `XGBoost`, `LightGBM`) pada 5 konfigurasi fitur ablasi (`M1` s.d. `M5`), mengalibrasi ambang keputusan $\tau_{\text{OOF}}$ secara *Temporal Out-of-Fold*, menjalankan uji signifikansi statistik **Newey-West HAC (**$L=2$**)**, serta merender **Gambar 3, Gambar 4, dan Gambar 8**. Menyimpan `stage04_state.pkl`. | 
| **`05_Deteksi_Dini_SHAP_dan_Prediksi_2026.ipynb`** | **Tahap 7, Tahap 8, Tahap 8B, Tahap 9, & Unit Tests** | Memuat `stage04_state.pkl`, menghitung *Early Warning Detection Rate* dan *Lead Time* terikat horizon $H=2$ (**Gambar 5**), mendekomposisi kontribusi fitur menggunakan **SHAP** (**Gambar 6**), menghasilkan peramalan prospektif Global (`2026-Q3`) dan **Mode Filter Regional Indonesia (`2026-Q4` tanpa 2022 pada Tahap 8B)** (**Gambar 7**), mengekspor artefak aplikasi web pada **Tahap 9**, serta mengeksekusi **37 Unit Tests** otomatis (`OK`). | 

## 5. Spesifikasi 4 Fitur Inti Multi-Indikator & Konfigurasi Ablasi (`M1`–`M5`)

Setiap observasi kandidat *Risk Set* $(s, t) \in \mathcal{R}_t$ direpresentasikan oleh vektor **4 fitur kausal** $\mathbf{x}_{s,t} \in \mathbb{R}^4$ yang dihitung murni dari histori hingga kuartal $t$:

| Kode Fitur | Nama Kolom pada Notebook & DB | Kelompok Sinyal | Rumus & Makna Operasional | Bobot SHAP (Global) | Bobot SHAP (Indonesia) | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **Fitur 01** | `qoq_growth_smoothed` | *Demand Dynamics* | Pertumbuhan pangsa pasar kuartal-ke-kuartal terhaluskan $\frac{p_{s,t} - p_{s,t-1}}{p_{s,t-1} + \epsilon}$ | `27,82%` | `44,12%` | 
| **Fitur 02** | `ema_ratio_short_long` | *Demand Dynamics* | Rasio akselerasi *Exponential Moving Average* jangka pendek terhadap jangka panjang $\frac{\text{EMA}_2(p_{s,t})}{\text{EMA}_4(p_{s,t}) + \epsilon}$ | `37,05%` | `34,36%` | 
| **Fitur 03** | `cooc_weighted_degree_roll4` | *Context Evolution* | Rerata bergerak 4 kuartal dari total bobot kemiripan Jaccard $\sum_{v \in \mathcal{N}_t(s)} J_t(s, v)$ pada graf ko-okurensi lowongan | `22,64%` | `11,00%` | 
| **Fitur 04** | `role_entropy_roll4` | *Role Expansion* | Rerata bergerak 4 kuartal dari entropi Shannon ternormalisasi $H_{s,t}^{(\text{role})} \in [0, 1]$ lintas 12 kategori peran TI terverifikasi | `12,49%` | `10,53%` | 

Desain studi ablasi menguji penambahan kelompok sinyal secara bertahap pada populasi kandidat *Risk Set* yang identik:

* **`M1_Demand_Only` (2 Fitur):** `["qoq_growth_smoothed", "ema_ratio_short_long"]` (*Baseline* dinamika permintaan murni).

* **`M2_Context_Only` (1 Fitur):** `["cooc_weighted_degree_roll4"]` (Sinyal konteks jaringan ko-okurensi tunggal).

* **`M3_Demand_Context` (3 Fitur):** `["qoq_growth_smoothed", "ema_ratio_short_long", "cooc_weighted_degree_roll4"]` (Permintaan + Evolusi Konteks).

* **`M4_Demand_Context_Role` / `M5_Full_Multi_Signal` (4 Fitur Inti Lengkap):** `["qoq_growth_smoothed", "ema_ratio_short_long", "cooc_weighted_degree_roll4", "role_entropy_roll4"]`.

## 6. Ringkasan Hasil Eksperimen Terverifikasi (Global vs Indonesia)

| Metrik Evaluasi Utama | Korpus Global (`2021-Q1` – `2026-Q3`) | Korpus Regional Indonesia (`2023-Q1` – `2026-Q4`, Tanpa 2022) | 
| ----- | ----- | ----- | 
| **PR-AUC (*Average Precision*) `M1` Baseline (RF)** | Val AP: `0,7030` · Holdout AP: `0,7163` · OOS AP: `0,7074` | Val AP: `0,6760` · Holdout AP: `0,6394` · OOS AP: `0,6577` | 
| **PR-AUC (*Average Precision*) `M3` Demand+Context (RF)** | Val AP: `0,7730` · Holdout AP: `0,7125` · OOS AP: `0,7529` | Val AP: `0,7175` · Holdout AP: `0,7121` · OOS AP: `0,7148` | 
| **PR-AUC (*Average Precision*) `M4/M5` 4 Fitur Inti (RF)** | **Val AP: `0,7789`** · **Holdout AP: `0,7280`** · **OOS AP: `0,7619`** | **Val AP: `0,7006`** · **Holdout AP: `0,7452` (+10,58% vs M1)** · **OOS AP: `0,7229`** | 
| **ROC-AUC & Precision@K (`M4/M5` RF)** | ROC-AUC: `0,7828` · P@5: `85,0%` · P@10: `81,7%` | ROC-AUC: `0,6964` · F1: `0,6678` · **P@5: `90,0%`** · **P@10: `78,8%`** | 
| **Uji Statistik Newey-West HAC (**$L=2$**)** | `M4 vs M1` (RF): $\Delta\text{AP} = +0,0542$, $t = 2,4276$, $p = 0,0168 < 0,05$ | `M3 vs M1` (RF): $t = 2,9368$, $p = 0,0109$ · `M4 vs M1` (XGB): $t = 2,7067$, $p = 0,0152 < 0,05$ | 
| **Tingkat Deteksi Dini (*Early Warning Rate*,** $H=2$**)** | **51 dari 69 skill** emerging terdeteksi dini (**`73,91%`**) | **48 dari 62 skill** emerging terdeteksi dini (**`77,42%`**) | 
| **Rerata *Lead Time* Peringatan Dini** | **`1,45 kuartal`** sebelum kemunculan (`1,67Q` ke puncak) | **`1,44 kuartal`** sebelum kemunculan (`1,52Q` ke puncak) | 

## 7. Arsitektur Aplikasi Web `SINYAL · EarlySkill-ID` & Basis Data SQLite3

Aplikasi web prototipe tersedia dalam dua kemasan:

1. **`SINYAL_EarlySkill_ID_Web_App.html` (Mandiri / *Standalone Single-File*):** Dapat dibuka langsung di peramban web atau Canvas tanpa instalasi server, karena seluruh bundel JSON *Dual-Corpus* (`INDONESIA` dan `GLOBAL`) telah disematkan di dalam memori antarmuka.

2. **`SINYAL_EarlySkill_ID_Fullstack_App.zip` (Paket *Fullstack* Python + SQLite3 + Model `.pkl`):** Memuat server backend `app.py`, basis data relasional `earlyskill_id.db` (Global) dan `earlyskill_id_indo.db` (Indonesia 2023–2026), objek model terlatih `trained_models_4fitur.pkl` dan `trained_models_indo_4fitur.pkl`, serta bundel JSON lengkap.

### Struktur 4 Halaman Antarmuka Web (`01` s.d. `04`)

* **Bilah Atas — Mode Filter Dataset (`🇮🇩 Pasar Indonesia 2023–2026` vs `🌐 Global Benchmark 2021–2026`):** Mengganti seluruh metrik KPI, daftar kuartal evaluasi/prospektif (`2026-Q4` pada Indonesia; `2026-Q3` pada Global), ambang kuantil kausal (`b_max`, `c_max`), *sparkline* trajektori, graf ko-okurensi, matriks ablasi, hingga bukti lowongan kerja sumber secara *real-time*.

* **Halaman `01 · Dashboard` (Watchlist & Permukaan Sinyal):** Menampilkan 4 kartu KPI utama, bilah penyaring interaktif (Kuartal, Domain TI, Model Ablasi, Minimum Lowongan dengan validasi *Boundary Value Analysis* `c_s,t >= 3`, Top-K `1..50`, sanitasi anti-injeksi pada kotak pencarian), tabel peringkat probabilitas prediksi $\hat{p}_{s,t}$ terhadap $\tau_{\text{OOF}}$, dan fitur ekspor CSV.

* **Halaman `02 · Eksplorasi sinyal` (4 Fitur, Ko-okurensi & Peran):** Menyajikan grafik *sparkline* ke-4 fitur inti dengan garis penanda kuartal peringatan dini pertama ($t_{\text{warn}}$), indikator posisi pangsa pasar terhadap ambang `b_max` dan `c_max`, sub-graf egosentris Top-5 pasangan ko-okurensi Jaccard, serta diagram batang pergeseran difusi lintas 12 kategori peran TI ($t-4$ vs $t$).

* **Halaman `03 · Evaluasi model` (Ablasi `M1`–`M5` & Walk-Forward):** Menampilkan *heatmap* interaktif PR-AUC 5 konfigurasi $\times$ 4 algoritma, rincian *Confusion Matrix* dan uji *Newey-West HAC*, distribusi *Early Warning Lead Time* ($1\text{Q}$ vs $2\text{Q}$ sebelum kemunculan), serta diagram skema *Expanding Walk-Forward Purging Gap* $H=2$.

* **Halaman `04 · SHAP & bukti` (Waterfall & Lowongan Sumber):** Menampilkan grafik *SHAP Waterfall* lokal skala probabilitas ($\hat{p}_{s,t} = \phi_0 + \sum_{k=1}^4 \phi_k$), perbandingan bobot SHAP global, serta tabel audit keterlacakan lowongan kerja sumber (`Job_Skill_Provenance`) yang dapat diklik per baris.

### Skema 6 Tabel Basis Data Relasional (`earlyskill_id.db` & `earlyskill_id_indo.db`)

Sesuai *Physical Data Schema* (Gambar 3.3 Bab III Proposal Skripsi), basis data SQLite3 yang diekspor oleh **Tahap 9 Notebook** mengimplementasikan 6 tabel fisik:

1. `Raw_Job_Postings`: Data lowongan kerja TI hasil prapemrosesan, deduplikasi, dan normalisasi peran.

2. `Job_Skill_Provenance`: Tabel relasi kemunculan setiap skill kanonis pada masing-masing `job_id` beserta kategori peran TI-nya.

3. `Skill_Taxonomy_Master`: Tabel referensi ke-113 keterampilan TI kanonis, kategori domain, dan daftar pola alias regex.

4. `Cooccurrence_Graph_Edges`: Tabel sisi graf ko-okurensi kuartalan (`skill_u`, `skill_v`, `cooc_count`, `jaccard_weight`, `cross_domain`).

5. `Skill_Quarter_Panel`: Tabel panel fitur kuartalan (`skill`, `year_quarter`, `skill_share`, ke-4 fitur inti, status `in_risk_set`, dan label `target_emerging`).

6. `Predictions_SHAP_LeadTime`: Tabel hasil prediksi probabilitas *Out-of-Sample* dan prospektif lintas model `M1`–`M5` beserta ambang $\tau_{\text{OOF}}$ dan peringkat kuartalan.

## 8. Cara Menjalankan Notebook & Aplikasi Web Secara Lokal

### A. Menjalankan Pipeline Jupyter Notebook

1. Pastikan berkas dataset `theirstack_pure_combined_2021_2026.csv` dan `indonesia_it_jobs_2023_2026.csv` berada pada direktori kerja proyek (atau di dalam subfolder `data/raw/`).

2. Untuk menjalankan keseluruhan eksperimen sekaligus (termasuk Mode Filter Regional Indonesia pada Tahap 8B, ekspor basis data/model pada Tahap 9, dan 37 pengujian unit), buka dan jalankan seluruh sel pada:

   ```
   jupyter notebook Emerging_IT_Skills_Full_Pipeline_4Fitur.ipynb
   
   ```

3. Untuk demonstrasi bertahap saat sidang skripsi, jalankan ke-5 notebook modular secara berurutan:

   * `01_Preprocessing_dan_Taksonomi_Skill.ipynb`

   * `02_Panel_Temporal_dan_Rekayasa_Fitur.ipynb`

   * `03_Pelabelan_Target_dan_Skema_Walk_Forward.ipynb`

   * `04_Evaluasi_Model_ML_dan_Studi_Ablasi.ipynb`

   * `05_Deteksi_Dini_SHAP_dan_Prediksi_2026.ipynb`

### B. Menjalankan Aplikasi Web `SINYAL · EarlySkill-ID`

1. **Mode Mandiri (*Standalone Browser*):** Buka berkas `SINYAL_EarlySkill_ID_Web_App.html` langsung menggunakan peramban web modern (Chrome, Edge, Firefox, atau Safari).

2. **Mode Server Lokal (*Fullstack Python + SQLite3 + Pickle REST API*):** Ekstrak arsip `SINYAL_EarlySkill_ID_Fullstack_App.zip`, masuk ke direktorinya, lalu jalankan:

   ```
   cd SINYAL_EarlySkill_ID_Fullstack
   python3 app.py
   
   ```

   Buka `http://localhost:8000` di peramban web Anda. Endpoint API lokal yang tersedia mencakup `/api/meta`, `/api/watchlist?quarter=2026-Q4`, dan `/api/predict?qoq_growth_smoothed=0.25&ema_ratio_short_long=1.18&cooc_weighted_degree_roll4=2.50&role_entropy_roll4=0.35`.