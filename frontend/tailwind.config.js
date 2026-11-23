/** Tailwind 3.4.16 config for the compiled frontend CSS (see "Frontend CSS" in README.md). */
module.exports = {
  content: ["./frontend/index.html", "./frontend/app.js"],
  theme: { extend: { fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"] } } },
};
