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

  $(".project-showcase").each(function () {
  const section = this;
  const list = section.querySelector(".project-list");
  if (!list) return;

  section.classList.add("enhanced");

  const wrap = document.createElement("div");
  const track = document.createElement("div");

  wrap.className = "project-track-wrap";
  track.className = "project-track";

  track.dataset.mouseDownAt = "0";
  track.dataset.prevPercentage = "0";
  track.dataset.percentage = "0";

  list.querySelectorAll("li").forEach(item => {
    const panel = document.createElement("a");
    panel.className = "project-panel";
    panel.href = item.dataset.url || "#";

    const image = document.createElement("img");
    image.className = "image";
    image.src = item.dataset.image;
    image.alt = "";
    image.draggable = false;

    panel.appendChild(image);
    track.appendChild(panel);
  });

  wrap.appendChild(track);
  list.replaceWith(wrap);

  let moved = false;

  const getMinPercentage = () => {
    const trackWidth = track.scrollWidth;
    const wrapWidth = wrap.clientWidth;

    if (trackWidth <= wrapWidth) return 0;

    return -((trackWidth - wrapWidth) / trackWidth) * 100;
  };

  const clamp = percentage => {
    const min = getMinPercentage();
    return Math.max(Math.min(percentage, 0), min);
  };

  const moveTo = (percentage, duration = 900) => {
    const nextPercentage = clamp(percentage);

    track.dataset.percentage = nextPercentage;

    track.animate(
      {
        transform: `translate(${nextPercentage}%, -50%)`
      },
      {
        duration,
        fill: "forwards",
        easing: "cubic-bezier(.22,.61,.36,1)"
      }
    );

    for (const image of track.getElementsByClassName("image")) {
      image.animate(
        {
          objectPosition: `${100 + nextPercentage}% center`
        },
        {
          duration,
          fill: "forwards",
          easing: "cubic-bezier(.22,.61,.36,1)"
        }
      );
    }
  };

  const handleDown = e => {
    moved = false;

    track.dataset.mouseDownAt = e.clientX;
    track.classList.add("is-dragging");

    track.setPointerCapture?.(e.pointerId);
  };

  const handleUp = e => {
    track.dataset.mouseDownAt = "0";
    track.dataset.prevPercentage = track.dataset.percentage || "0";
    track.classList.remove("is-dragging");

    if (track.hasPointerCapture?.(e.pointerId)) {
      track.releasePointerCapture(e.pointerId);
    }
  };

  const handleMove = e => {
    if (track.dataset.mouseDownAt === "0") return;

    const mouseDownAt = parseFloat(track.dataset.mouseDownAt);
    const prevPercentage = parseFloat(track.dataset.prevPercentage || "0");

    const mouseDelta = mouseDownAt - e.clientX;
    const maxDelta = window.innerWidth / 2;

    if (Math.abs(mouseDelta) > 5) moved = true;

    const percentage = (mouseDelta / maxDelta) * -100;
    const nextPercentage = prevPercentage + percentage;

    moveTo(nextPercentage, 900);
  };

  const handleWheel = e => {
    e.preventDefault();

    const currentPercentage = parseFloat(track.dataset.percentage || "0");

    /*
      Use whichever wheel direction is stronger.
      This gives good support for normal mouse wheels,
      trackpads, and horizontal scrolling gestures.
    */
    const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY)
      ? e.deltaX
      : e.deltaY;

    const nextPercentage =
      currentPercentage + (delta / window.innerWidth) * -100;

    track.dataset.prevPercentage = nextPercentage;

    moveTo(nextPercentage, 700);
  };

  track.addEventListener("pointerdown", handleDown);
  track.addEventListener("pointerup", handleUp);
  track.addEventListener("pointercancel", handleUp);
  track.addEventListener("pointermove", handleMove);

  wrap.addEventListener("wheel", handleWheel, { passive: false });

  track.addEventListener("click", e => {
    if (moved) {
      e.preventDefault();
    }
  });

  track.addEventListener("dragstart", e => {
    e.preventDefault();
  });

  window.addEventListener("resize", () => {
    moveTo(parseFloat(track.dataset.percentage || "0"), 0);
    track.dataset.prevPercentage = track.dataset.percentage || "0";
  });
});

});