document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("predictForm");
  const resultBox = document.querySelector(".result-box");

  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    const formData = new FormData(form);

    resultBox.style.display = "block";
    resultBox.innerHTML = "<p>⏳ Detecting... please wait...</p>";

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData
      });

      const html = await response.text();

      // Parse result from returned HTML
      const parser = new DOMParser();
      const doc = parser.parseFromString(html, "text/html");
      const newResult = doc.querySelector(".result-box");

      if (newResult) {
        resultBox.innerHTML = newResult.innerHTML;
      } else {
        resultBox.innerHTML = "<p>⚠️ Error: No result returned.</p>";
      }
    } catch (err) {
      console.error(err);
      resultBox.innerHTML = "<p>❌ An error occurred during detection.</p>";
    }
  });
});
