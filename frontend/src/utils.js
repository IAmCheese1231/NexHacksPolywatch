export function prettyProb(p) {
  if (typeof p !== "number") return "";
  return `${(p * 100).toFixed(2)}%`;
}
