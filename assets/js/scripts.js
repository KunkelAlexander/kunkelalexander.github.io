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
  const projectTrack = document.getElementById("project-track");

  if (projectTrack) {
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

    const clampPercentage = value => {
      return Math.max(Math.min(value, 0), -75);
    };

    const handleOnDown = e => {
      projectTrack.dataset.mouseDownAt = e.clientX;
    };

    const handleOnUp = () => {
      projectTrack.dataset.mouseDownAt = "0";
      projectTrack.dataset.prevPercentage =
        projectTrack.dataset.percentage || "0";
    };

    const handleOnMove = e => {
      if (projectTrack.dataset.mouseDownAt === "0") return;

      const mouseDelta =
        parseFloat(projectTrack.dataset.mouseDownAt) - e.clientX;

      const maxDelta = window.innerWidth / 2;

      const percentage = (mouseDelta / maxDelta) * -100;

      const nextPercentageUnconstrained =
        parseFloat(projectTrack.dataset.prevPercentage || "0") + percentage;

      const nextPercentage = clampPercentage(nextPercentageUnconstrained);

      animateTrack(nextPercentage, 1200);
    };

    const handleOnWheel = e => {
      e.preventDefault();

      const scrollSpeed = 0.8;
      const delta = e.deltaY * scrollSpeed;
      const maxDelta = window.innerWidth / 2;

      const percentage = (delta / maxDelta) * -100;

      const currentPercentage =
        parseFloat(projectTrack.dataset.percentage || "0");

      const nextPercentage = clampPercentage(
        currentPercentage + percentage
      );

      projectTrack.dataset.prevPercentage = nextPercentage;

      animateTrack(nextPercentage, 600);
    };

    window.addEventListener("mousedown", handleOnDown);
    window.addEventListener("mouseup", handleOnUp);
    window.addEventListener("mousemove", handleOnMove);

    window.addEventListener("touchstart", e => handleOnDown(e.touches[0]));
    window.addEventListener("touchend", handleOnUp);
    window.addEventListener("touchmove", e => handleOnMove(e.touches[0]));

    projectTrack.addEventListener("wheel", handleOnWheel, {
      passive: false
    });
  }

});