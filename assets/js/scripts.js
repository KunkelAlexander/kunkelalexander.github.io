// A $( document ).ready() block.
$(document).ready(function () {

  // DropCap.js
  var dropcaps = document.querySelectorAll(".dropcap");
  window.Dropcap.layout(dropcaps, 2);

  // Responsive-Nav
  var nav = responsiveNav(".nav-collapse");

  // Round Reading Time
  $(".time").text(function (index, value) {
    return Math.round(parseFloat(value));
  });

  // Project showcase
  // Project showcase
const projectTrack = document.getElementById("project-track");

if (projectTrack) {
  let isDragging = false;
  let hasMoved = false;

  const clampPercentage = value => Math.max(Math.min(value, 0), -90);

  const animateTrack = (nextPercentage, duration = 1200) => {
    projectTrack.dataset.percentage = nextPercentage;

    projectTrack.animate(
      { transform: `translate(${nextPercentage}%, -50%)` },
      { duration: duration, fill: "forwards" }
    );

    for (const image of projectTrack.getElementsByClassName("image")) {
      image.animate(
        { objectPosition: `${100 + nextPercentage}% center` },
        { duration: duration, fill: "forwards" }
      );
    }
  };

  projectTrack.addEventListener("pointerdown", e => {
    isDragging = true;
    hasMoved = false;

    projectTrack.dataset.mouseDownAt = e.clientX;
    projectTrack.setPointerCapture(e.pointerId);
  });

  projectTrack.addEventListener("pointerup", e => {
    isDragging = false;
    projectTrack.dataset.mouseDownAt = "0";
    projectTrack.dataset.prevPercentage = projectTrack.dataset.percentage || "0";

    if (projectTrack.hasPointerCapture(e.pointerId)) {
      projectTrack.releasePointerCapture(e.pointerId);
    }
  });

  projectTrack.addEventListener("pointercancel", e => {
    isDragging = false;
    projectTrack.dataset.mouseDownAt = "0";

    if (projectTrack.hasPointerCapture(e.pointerId)) {
      projectTrack.releasePointerCapture(e.pointerId);
    }
  });

  projectTrack.addEventListener("pointermove", e => {
    if (!isDragging) return;

    const mouseDelta = parseFloat(projectTrack.dataset.mouseDownAt) - e.clientX;

    if (Math.abs(mouseDelta) > 5) {
      hasMoved = true;
    }

    const maxDelta = window.innerWidth / 2;
    const percentage = (mouseDelta / maxDelta) * -100;

    const nextPercentageUnconstrained =
      parseFloat(projectTrack.dataset.prevPercentage || "0") + percentage;

    const nextPercentage = clampPercentage(nextPercentageUnconstrained);

    animateTrack(nextPercentage, 1200);
  });

  projectTrack.addEventListener("click", e => {
    if (hasMoved) {
      e.preventDefault();
      e.stopPropagation();
    }
  });

  projectTrack.addEventListener("wheel", e => {
    e.preventDefault();

    const scrollSpeed = 0.8;
    const delta = e.deltaY * scrollSpeed;
    const maxDelta = window.innerWidth / 2;

    const percentage = (delta / maxDelta) * -100;
    const currentPercentage = parseFloat(projectTrack.dataset.percentage || "0");

    const nextPercentage = clampPercentage(currentPercentage + percentage);

    projectTrack.dataset.prevPercentage = nextPercentage;

    animateTrack(nextPercentage, 600);
  }, { passive: false });
}

});