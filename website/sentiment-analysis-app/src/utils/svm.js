/**
 * SVM (Support Vector Machine) untuk klasifikasi sentimen teks.
 * Machine Learning klasik — Linear SVM dengan SGD (hinge loss), BUKAN deep learning.
 * Multi-class: One-vs-Rest (satu classifier biner per kelas).
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

function buildVocab(data) {
  const vocab = new Map();
  let idx = 0;
  data.forEach((row) => {
    tokenize(row.Judul).forEach((w) => {
      if (!vocab.has(w)) vocab.set(w, idx++);
    });
  });
  return vocab;
}

function textToVector(tokens, vocab, dim) {
  const vec = new Float64Array(dim);
  tokens.forEach((w) => {
    const i = vocab.get(w);
    if (i !== undefined) vec[i] += 1;
  });
  let norm = 0;
  for (let j = 0; j < dim; j++) norm += vec[j] * vec[j];
  norm = Math.sqrt(norm) || 1;
  for (let j = 0; j < dim; j++) vec[j] /= norm;
  return vec;
}

/**
 * Linear SVM biner: SGD dengan hinge loss, L2 regularization.
 * y in {-1, +1}, predict: sign(w·x + b)
 */
function trainBinarySVM(X, y, options = {}) {
  const { epochs = 20, lr = 0.01, C = 0.1 } = options;
  const n = X.length;
  const dim = X[0].length;
  const w = new Float64Array(dim);
  let b = 0;

  for (let ep = 0; ep < epochs; ep++) {
    for (let i = 0; i < n; i++) {
      const x = X[i];
      const yi = y[i];
      let score = b;
      for (let j = 0; j < dim; j++) score += w[j] * x[j];
      const margin = yi * score;
      if (margin < 1) {
        for (let j = 0; j < dim; j++) w[j] += lr * (yi * x[j] - C * w[j]);
        b += lr * yi;
      } else {
        for (let j = 0; j < dim; j++) w[j] -= lr * C * w[j];
      }
    }
  }

  return {
    w,
    b,
    predict(x) {
      let score = b;
      for (let j = 0; j < dim; j++) score += w[j] * x[j];
      return score;
    },
  };
}

/**
 * Latih model SVM (One-vs-Rest) dari data: array of { Judul, Sentimen }
 */
export function trainSVM(data) {
  const vocab = buildVocab(data);
  const dim = vocab.size;
  if (dim === 0) return null;

  const tokensList = data.map((row) => tokenize(row.Judul));
  const X = tokensList.map((tokens) => textToVector(tokens, vocab, dim));

  const classToIdx = { Positif: 0, Negatif: 1, Netral: 2 };
  const classifiers = [];

  SENTIMEN_CLASSES.forEach((c, idx) => {
    const y = data.map((row) => (row.Sentimen === c ? 1 : -1));
    const svm = trainBinarySVM(X, y, { epochs: 25, lr: 0.05, C: 0.01 });
    classifiers.push(svm);
  });

  function softmax(scores) {
    const exp = scores.map((s) => Math.exp(s - Math.max(...scores)));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map((e) => e / sum);
  }

  return {
    vocab,
    dim,
    classifiers,
    predict(text) {
      const tokens = tokenize(text);
      if (tokens.length === 0) return { label: 'Netral', scores: { Positif: 0.33, Negatif: 0.33, Netral: 0.34 } };

      const x = textToVector(tokens, vocab, dim);
      const rawScores = classifiers.map((svm) => svm.predict(x));
      const scoresArr = softmax(rawScores);
      const scores = {};
      SENTIMEN_CLASSES.forEach((c, i) => {
        scores[c] = scoresArr[i];
      });
      const bestIdx = scoresArr.indexOf(Math.max(...scoresArr));
      const label = SENTIMEN_CLASSES[bestIdx];
      return { label, scores };
    },
  };
}
