import { useState, useEffect, useMemo } from "react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";
import {
  parseCSV,
  parseCSVUpload,
  getSentimentStats,
  getSentimentByYear,
} from "./utils/csvParser";
import "./App.css";

const SENTIMENT_FILTERS = ["Semua", "Positif", "Negatif", "Netral"];

function App() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState("");
  const [sentimentFilter, setSentimentFilter] = useState("Semua");

  const [inputText, setInputText] = useState("");
  const [predictionResult, setPredictionResult] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);

  // Load Data CSV Awal (VERSI DEBUGGING)
  useEffect(() => {
    fetch("/dataset_labeled.csv")
      .then((res) => {
        if (!res.ok) throw new Error("Gagal memuat dataset HTTP Error");
        return res.text();
      })
      .then((text) => {
        // CCTV 1: Cek isi teks aslinya
        console.log(
          "CCTV 1 - ISI TEKS ASLI (100 huruf pertama):",
          text.substring(0, 100),
        );

        // Kalau teksnya diawali dengan <!DOCTYPE html> atau <html, berarti file CSV tidak ditemukan!
        if (text.trim().startsWith("<")) {
          throw new Error(
            "Yang terbaca malah file HTML! Pastikan file dataset_labeled.csv benar-benar diletakkan di dalam folder: public/assets/",
          );
        }

        const rows = parseCSV(text);

        // CCTV 2: Cek hasil bacanya
        console.log("CCTV 2 - HASIL BARIS PERTAMA:", rows[0]);

        setData(rows);
        setError(null);
      })
      .catch((err) => {
        console.error("ERROR DARI CATCH:", err);
        setError(err.message);
      })
      .finally(() => setLoading(false));
  }, []);

  const stats = useMemo(() => getSentimentStats(data), [data]);
  const byYear = useMemo(() => getSentimentByYear(data), [data]);

  const filteredData = useMemo(() => {
    return data.filter((row) => {
      const matchSearch =
        !search ||
        row.Judul.toLowerCase().includes(search.toLowerCase()) ||
        row.Tanggal.includes(search);
      // Bikin case-insensitive biar nggak error kalau huruf besar/kecil beda
      const sentimenBawaan = row.Sentimen
        ? row.Sentimen.toLowerCase().trim()
        : "";
      const matchSentiment =
        sentimentFilter === "Semua" ||
        sentimenBawaan === sentimentFilter.toLowerCase();
      return matchSearch && matchSentiment;
    });
  }, [data, search, sentimentFilter]);

  // FIX MASALAH ANGKA 0: Bikin perbandingan hurufnya kebal huruf besar/kecil (toLowerCase)
  const countPositif = data.filter(
    (r) => r.Sentimen && r.Sentimen.toLowerCase().trim() === "positif",
  ).length;
  const countNegatif = data.filter(
    (r) => r.Sentimen && r.Sentimen.toLowerCase().trim() === "negatif",
  ).length;
  const countNetral = data.filter(
    (r) => r.Sentimen && r.Sentimen.toLowerCase().trim() === "netral",
  ).length;

  const handlePredict = async () => {
    if (!inputText.trim()) return;

    setIsPredicting(true);
    setPredictionResult(null);

    try {
      const response = await fetch(
        "https://kaelgogo-api-sentimen-finansial.hf.space/api/analisis",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ teks: inputText }),
        },
      );

      if (!response.ok) throw new Error("Server AI belum nyala");

      const resData = await response.json();
      setPredictionResult(resData.sentimen);
    } catch (err) {
      console.error(err);
      alert(
        "Gagal konek ke AI! Pastikan uvicorn main:app sudah jalan di terminal.",
      );
    } finally {
      setIsPredicting(false);
    }
  };

  if (loading) {
    return (
      <div className="app">
        <div className="loading">
          <p>Memuat data sentiment...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Sentiment Analysis Berita Publik</h1>
        <p>
          Dashboard Analitik & Live Prediksi AI menggunakan Support Vector
          Machine (SVM)
        </p>
      </header>

      {/* Tampilan Error jika file CSV tidak ketemu */}
      {error && (
        <div
          style={{
            padding: "15px",
            backgroundColor: "#ef4444",
            color: "white",
            borderRadius: "8px",
            marginBottom: "20px",
          }}
        >
          <strong>Error: </strong> {error}
        </div>
      )}

      {/* SECTION PREDIKSI YANG SUDAH DIPERBAIKI WARNANYA */}
      <section
        className="upload-section"
        style={{
          border: "1px solid var(--border)",
          backgroundColor: "var(--bg-secondary)",
        }}
      >
        <h3 style={{ margin: "0 0 10px 0" }}>
          ✨ Coba Langsung Model AI (Real-time)
        </h3>
        <p className="upload-desc">
          Ketik judul berita baru, AI kita (SVM) akan menebak sentimennya
          langsung dari server Python.
        </p>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "10px",
            marginTop: "15px",
          }}
        >
          <textarea
            rows="3"
            className="search-input"
            placeholder="Masukkan judul berita di sini (Misal: 'Pemerintah sukses bangun jalan tol baru')"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            style={{
              width: "100%",
              resize: "vertical",
              backgroundColor: "var(--bg-primary)",
              color: "var(--text-primary)",
            }}
          />
          <button
            className="filter-btn active"
            onClick={handlePredict}
            disabled={isPredicting || !inputText.trim()}
            style={{
              padding: "10px 20px",
              alignSelf: "flex-start",
              cursor: isPredicting ? "not-allowed" : "pointer",
              fontWeight: "bold",
            }}
          >
            {isPredicting ? "AI Sedang Berpikir..." : "Analisis Teks"}
          </button>

          {predictionResult && (
            <div
              style={{
                marginTop: "15px",
                padding: "15px",
                background: "var(--bg-primary)",
                borderRadius: "8px",
                border: "1px solid var(--border)",
              }}
            >
              <strong style={{ fontSize: "1.1rem" }}>
                Hasil Prediksi Model SVM:{" "}
              </strong>
              <span
                className={`badge ${predictionResult.toLowerCase()}`}
                style={{
                  fontSize: "1.2rem",
                  padding: "5px 15px",
                  marginLeft: "10px",
                }}
              >
                {predictionResult}
              </span>
            </div>
          )}
        </div>
      </section>

      {/* DASHBOARD STATISTIK */}
      <div className="stats-grid" style={{ marginTop: "30px" }}>
        <div className="stat-card positive">
          <div className="value">{countPositif.toLocaleString("id-ID")}</div>
          <div className="label">Positif</div>
        </div>
        <div className="stat-card negative">
          <div className="value">{countNegatif.toLocaleString("id-ID")}</div>
          <div className="label">Negatif</div>
        </div>
        <div className="stat-card neutral">
          <div className="value">{countNetral.toLocaleString("id-ID")}</div>
          <div className="label">Netral</div>
        </div>
        <div className="stat-card">
          <div className="value">{data.length.toLocaleString("id-ID")}</div>
          <div className="label">Total Dataset</div>
        </div>
      </div>

      <div className="charts-row">
        <div className="chart-card">
          <h3>Distribusi Sentimen</h3>
          <div className="pie-container">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={stats}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                  nameKey="name"
                  label={({ name, percent }) =>
                    `${name} ${(percent * 100).toFixed(0)}%`
                  }
                >
                  {stats.map((entry, index) => (
                    <Cell key={index} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value) => [
                    value.toLocaleString("id-ID"),
                    "Jumlah",
                  ]}
                  contentStyle={{
                    background: "var(--bg-secondary)",
                    border: "1px solid var(--border)",
                    borderRadius: "8px",
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-card">
          <h3>Sentimen per Tahun</h3>
          <div style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={byYear}
                margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis
                  dataKey="year"
                  stroke="var(--text-secondary)"
                  fontSize={12}
                />
                <YAxis stroke="var(--text-secondary)" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    background: "var(--bg-secondary)",
                    border: "1px solid var(--border)",
                    borderRadius: "8px",
                  }}
                />
                <Legend />
                <Bar
                  dataKey="Positif"
                  stackId="a"
                  fill="#22c55e"
                  name="Positif"
                />
                <Bar
                  dataKey="Netral"
                  stackId="a"
                  fill="#94a3b8"
                  name="Netral"
                />
                <Bar
                  dataKey="Negatif"
                  stackId="a"
                  fill="#ef4444"
                  name="Negatif"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <section>
        <h3
          style={{
            margin: "0 0 0.75rem 0",
            fontSize: "1rem",
            fontWeight: 600,
            color: "var(--text-secondary)",
          }}
        >
          Daftar Berita (Dataset Training)
        </h3>
        <div className="filters">
          <input
            type="text"
            className="search-input"
            placeholder="Cari judul atau tanggal..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          {SENTIMENT_FILTERS.map((f) => (
            <button
              key={f}
              className={`filter-btn ${sentimentFilter === f ? "active" : ""}`}
              onClick={() => setSentimentFilter(f)}
            >
              {f}
            </button>
          ))}
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>No</th>
                <th>Tanggal</th>
                <th>Judul</th>
                <th>Sentimen</th>
              </tr>
            </thead>
            <tbody>
              {filteredData.slice(0, 100).map((row, idx) => (
                <tr key={idx}>
                  <td>{idx + 1}</td>
                  <td>{row.Tanggal}</td>
                  <td className="judul-cell">
                    {row.Judul.length > 80
                      ? row.Judul.slice(0, 80) + "…"
                      : row.Judul}
                  </td>
                  <td>
                    <span
                      className={`badge ${row.Sentimen ? row.Sentimen.toLowerCase().trim() : "netral"}`}
                    >
                      {row.Sentimen}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

export default App;
