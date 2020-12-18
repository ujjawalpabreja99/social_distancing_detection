const fileInput = document.querySelector(".custom-file-input");

fileInput.addEventListener("change", (event) => {
  const fileLabel = event.target.nextElementSibling;

  const files = document.getElementById("video").files;

  if (!files[0]) {
    fileLabel.innerText = "Choose file";
  } else {
    fileLabel.innerText = files[0].name;
  }
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
