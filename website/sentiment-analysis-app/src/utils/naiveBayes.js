/**
 * Naive Bayes classifier untuk klasifikasi sentimen teks (Machine Learning klasik, BUKAN deep learning).
 * Model: Multinomial Naive Bayes dengan smoothing Laplace.
 */

const SENTIMEN_CLASSES = ['Positif', 'Negatif', 'Netral'];

function tokenize(text) {
  if (!text || typeof text !== 'string') return [];
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter((t) => t.length > 1);
}

/**
 * Latih model Naive Bayes dari data: array of { Judul, Sentimen }
 */
export function trainNaiveBayes(data) {
  const classCount = { Positif: 0, Negatif: 0, Netral: 0 };
  const wordCount = { Positif: {}, Negatif: {}, Netral: {} };
  const vocab = new Set();

  data.forEach((row) => {
    const label = row.Sentimen;
    if (!SENTIMEN_CLASSES.includes(label)) return;
    classCount[label]++;
    const tokens = tokenize(row.Judul);
    tokens.forEach((w) => {
      vocab.add(w);
      wordCount[label][w] = (wordCount[label][w] || 0) + 1;
    });
  });

  const totalDocs = data.length;
  const classPrior = {};
  const totalWordsPerClass = {};
  SENTIMEN_CLASSES.forEach((c) => {
    classPrior[c] = (classCount[c] + 1) / (totalDocs + SENTIMEN_CLASSES.length);
    totalWordsPerClass[c] = Object.values(wordCount[c]).reduce((a, b) => a + b, 0);
  });

  const vocabSize = vocab.size;
  return {
    classPrior,
    wordCount,
    classCount,
    vocabSize,
    predict(text) {
      const tokens = tokenize(text);
      if (tokens.length === 0) return { label: 'Netral', scores: { Positif: 0.33, Negatif: 0.33, Netral: 0.34 } };

      let bestLabel = 'Netral';
      let bestScore = -Infinity;
      const scores = {};

      SENTIMEN_CLASSES.forEach((c) => {
        let logProb = Math.log(classPrior[c]);
        const denom = totalWordsPerClass[c] + vocabSize;
        tokens.forEach((w) => {
          const count = (wordCount[c][w] || 0) + 1;
          logProb += Math.log(count / denom);
        });
        scores[c] = logProb;
        if (logProb > bestScore) {
          bestScore = logProb;
          bestLabel = c;
        }
      });

      const maxS = Math.max(...Object.values(scores));
      const expScores = {};
      SENTIMEN_CLASSES.forEach((c) => {
        expScores[c] = Math.exp(scores[c] - maxS);
      });
      const sumExp = Object.values(expScores).reduce((a, b) => a + b, 0);
      SENTIMEN_CLASSES.forEach((c) => {
        expScores[c] = expScores[c] / sumExp;
      });

      return { label: bestLabel, scores: expScores };
    },
  };
}
