const fileInput = document.querySelector(".custom-file-input");

fileInput.addEventListener("change", (event) => {
  const fileName = document.getElementById("video").files[0].name;
  const fileLabel = event.target.nextElementSibling;
  fileLabel.innerText = fileName;
});

const form = document.getElementById("form");

form.addEventListener("submit", () => {
  const dataset = document.getElementById("dataset").value;
  const video = document.getElementById("video").value;

  if (!video) {
    event.preventDefault();
    form.classList.add("was-validated");
    return false;
  }
  if (!dataset) {
    event.preventDefault();
    form.classList.add("was-validated");
    return false;
  }

  const loader = document.getElementById("loader");
  loader.classList.add("remove-cover");
  return true;
});
