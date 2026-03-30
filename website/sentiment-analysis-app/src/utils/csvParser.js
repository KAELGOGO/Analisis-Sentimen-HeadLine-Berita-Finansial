/**
 * Parse CSV string with support for quoted fields (handles commas inside quotes)
 */
export function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = parseCSVLine(lines[0]);
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVLine(lines[i]);
    if (values.length >= 4) {
      rows.push({
        Tanggal: values[0],
        Judul: values[1],
        URL: values[2],
        Sentimen: values[3].trim(),
      });
    }
  }
  return rows;
}

/**
 * Parse CSV upload: minimal 2 kolom (Tanggal, Judul), optional URL & Sentimen.
 * Untuk file yang akan dicek/dianalisis (boleh ada atau tidak kolom Sentimen).
 */
export function parseCSVUpload(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVLine(lines[i]);
    if (values.length >= 2) {
      rows.push({
        Tanggal: values[0] || '',
        Judul: values[1] || '',
        URL: values[2] || '',
        Sentimen: (values[3] || '').trim(),
      });
    }
  }
  return rows;
}

function parseCSVLine(line) {
  const result = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      inQuotes = !inQuotes;
    } else if (inQuotes) {
      current += c;
    } else if (c === ',') {
      result.push(current.trim());
      current = '';
    } else {
      current += c;
    }
  }
  result.push(current.trim());
  return result;
}

export function getSentimentStats(data) {
  const counts = { Positif: 0, Negatif: 0, Netral: 0 };
  data.forEach((row) => {
    const s = row.Sentimen;
    if (counts[s] !== undefined) counts[s]++;
  });
  return [
    { name: 'Positif', value: counts.Positif, fill: '#22c55e' },
    { name: 'Negatif', value: counts.Negatif, fill: '#ef4444' },
    { name: 'Netral', value: counts.Netral, fill: '#94a3b8' },
  ].filter((d) => d.value > 0);
}

export function getSentimentByYear(data) {
  const byYear = {};
  data.forEach((row) => {
    const match = row.Tanggal && row.Tanggal.match(/(\d{2})-(\d{2})-(\d{4})/);
    const year = match ? match[3] : 'Unknown';
    if (!byYear[year]) byYear[year] = { Positif: 0, Negatif: 0, Netral: 0 };
    const s = row.Sentimen;
    if (byYear[year][s] !== undefined) byYear[year][s]++;
  });
  return Object.entries(byYear)
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([year, v]) => ({
      year,
      Positif: v.Positif,
      Negatif: v.Negatif,
      Netral: v.Netral,
      total: v.Positif + v.Negatif + v.Netral,
    }));
}
