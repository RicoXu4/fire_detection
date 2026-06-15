const fileInput = document.querySelector("#fileInput");
const dropZone = document.querySelector("#dropZone");
const runButton = document.querySelector("#runButton");
const sourceImage = document.querySelector("#sourceImage");
const resultImage = document.querySelector("#resultImage");
const emptyResult = document.querySelector("#emptyResult");
const fireConfidence = document.querySelector("#fireConfidence");
const fireConfidenceValue = document.querySelector("#fireConfidenceValue");
const smokeConfidence = document.querySelector("#smokeConfidence");
const smokeConfidenceValue = document.querySelector("#smokeConfidenceValue");
const labelMode = document.querySelector("#labelMode");
const count = document.querySelector("#count");
const dimensions = document.querySelector("#dimensions");
const statusText = document.querySelector("#status");
const detectionsBody = document.querySelector("#detectionsBody");

let currentFile = null;

function setStatus(text) {
  statusText.textContent = text;
}

function setImage(img, src) {
  img.src = src;
  img.classList.add("hasImage");
}

function resetResults() {
  resultImage.removeAttribute("src");
  resultImage.classList.remove("hasImage");
  emptyResult.style.display = "block";
  count.textContent = "0";
  dimensions.textContent = "-";
  detectionsBody.innerHTML = '<tr><td colspan="3" class="muted">暂无检测结果</td></tr>';
}

function acceptFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("请选择图片");
    return;
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = () => setImage(sourceImage, reader.result);
  reader.readAsDataURL(file);
  runButton.disabled = false;
  resetResults();
  setStatus("就绪");
}

function renderDetections(data) {
  count.textContent = String(data.detections.length);
  dimensions.textContent = `${data.image_width} x ${data.image_height}`;

  if (data.visualization) {
    setImage(resultImage, `data:image/png;base64,${data.visualization}`);
    emptyResult.style.display = "none";
  }

  if (!data.detections.length) {
    detectionsBody.innerHTML = '<tr><td colspan="3" class="muted">未检测到目标</td></tr>';
    setStatus("完成");
    return;
  }

  setStatus(data.detections.some((det) => det.label === "fire" || det.label === "smoke") ? "有火情风险" : "完成");

  detectionsBody.innerHTML = data.detections
    .map((det) => {
      const label = det.label || "unknown";
      const box = det.box;
      const coords = [box.x1, box.y1, box.x2, box.y2].map((value) => Math.round(value));
      return `
        <tr>
          <td><span class="pill ${label}">${label}</span></td>
          <td>${det.confidence.toFixed(3)}</td>
          <td>${coords.join(", ")}</td>
        </tr>
      `;
    })
    .join("");
}

async function runDetection() {
  if (!currentFile) return;

  runButton.disabled = true;
  setStatus("检测中");

  const form = new FormData();
  form.append("file", currentFile);

  try {
    const params = new URLSearchParams({
      visualize: "true",
      fire_conf: fireConfidence?.value || "0.12",
      smoke_conf: smokeConfidence?.value || "0.30",
      label_mode: labelMode?.value || "risk",
    });
    const response = await fetch(`/detect?${params}`, {
      method: "POST",
      body: form,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "检测失败");
    }
    renderDetections(data);
  } catch (error) {
    setStatus(error.message);
  } finally {
    runButton.disabled = false;
  }
}

fileInput.addEventListener("change", (event) => {
  acceptFile(event.target.files[0]);
});

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("isDragging");
});
dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("isDragging");
});
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("isDragging");
  acceptFile(event.dataTransfer.files[0]);
});

runButton.addEventListener("click", runDetection);
fireConfidence?.addEventListener("input", () => {
  fireConfidenceValue.textContent = Number(fireConfidence.value).toFixed(2);
});

smokeConfidence?.addEventListener("input", () => {
  smokeConfidenceValue.textContent = Number(smokeConfidence.value).toFixed(2);
});
