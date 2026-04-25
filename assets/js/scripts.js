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
    track.innerHTML += `
      <a class="project-card" href="${item.dataset.url}">
        <img class="image" src="${item.dataset.image}" draggable="false" alt="">
        <span>
          <strong>${item.dataset.title}</strong>
          <small>${item.dataset.text}</small>
        </span>
      </a>`;
  });

  wrap.appendChild(track);
  list.replaceWith(wrap);

  let moved = false;
  let activePointerId = null;

  const getCardCenters = () => {
    const cards = track.querySelectorAll(".project-card");

    if (!cards.length) {
      return {
        firstCenter: 0,
        lastCenter: 0,
        trackWidth: 1
      };
    }

    const first = cards[0];
    const last = cards[cards.length - 1];

    return {
      firstCenter: first.offsetLeft + first.offsetWidth / 2,
      lastCenter: last.offsetLeft + last.offsetWidth / 2,
      trackWidth: track.scrollWidth
    };
  };

  const getBounds = () => {
    const { firstCenter, lastCenter, trackWidth } = getCardCenters();

    return {
      maxPct: (-firstCenter / trackWidth) * 100,
      minPct: (-lastCenter / trackWidth) * 100
    };
  };

  const clamp = pct => {
    const { minPct, maxPct } = getBounds();
    return Math.max(Math.min(pct, maxPct), minPct);
  };

  const setPosition = (pct, animate = true) => {
    const nextPct = clamp(pct);

    track.dataset.percentage = nextPct;

    if (animate) {
      track.animate(
        {
          transform: `translate(${nextPct}%, -50%)`
        },
        {
          duration: 1200,
          fill: "forwards"
        }
      );

      for (const img of track.getElementsByClassName("image")) {
        img.animate(
          {
            objectPosition: `${100 + nextPct}% center`
          },
          {
            duration: 1200,
            fill: "forwards"
          }
        );
      }
    } else {
      track.style.transform = `translate(${nextPct}%, -50%)`;

      for (const img of track.getElementsByClassName("image")) {
        img.style.objectPosition = `${100 + nextPct}% center`;
      }
    }
  };

  const centerFirstCard = () => {
    const { maxPct } = getBounds();

    track.dataset.prevPercentage = maxPct;
    track.dataset.percentage = maxPct;

    setPosition(maxPct, false);
  };

  requestAnimationFrame(centerFirstCard);

  window.addEventListener("resize", () => {
    const currentPct = parseFloat(track.dataset.percentage || "0");
    setPosition(currentPct, false);
    track.dataset.prevPercentage = track.dataset.percentage || "0";
  });

  wrap.addEventListener("pointerdown", e => {
    moved = false;
    activePointerId = e.pointerId;

    track.dataset.mouseDownAt = e.clientX;
    track.dataset.prevPercentage = track.dataset.percentage || "0";

    wrap.setPointerCapture(e.pointerId);
  });

  wrap.addEventListener("pointerup", e => {
    if (activePointerId !== e.pointerId) return;

    track.dataset.mouseDownAt = "0";
    track.dataset.prevPercentage = track.dataset.percentage || "0";

    if (wrap.hasPointerCapture(e.pointerId)) {
      wrap.releasePointerCapture(e.pointerId);
    }

    activePointerId = null;
  });

  wrap.addEventListener("pointercancel", e => {
    if (activePointerId !== e.pointerId) return;

    track.dataset.mouseDownAt = "0";
    track.dataset.prevPercentage = track.dataset.percentage || "0";

    activePointerId = null;
  });

  wrap.addEventListener("pointermove", e => {
    if (activePointerId !== e.pointerId) return;
    if (track.dataset.mouseDownAt === "0") return;

    const mouseDelta = parseFloat(track.dataset.mouseDownAt) - e.clientX;
    const maxDelta = window.innerWidth / 2;

    if (Math.abs(mouseDelta) > 5) {
      moved = true;
    }

    const percentage = (mouseDelta / maxDelta) * -100;
    const nextPercentage =
      parseFloat(track.dataset.prevPercentage || "0") + percentage;

    setPosition(nextPercentage);
  });

  wrap.addEventListener(
    "wheel",
    e => {
      e.preventDefault();

      const currentPct = parseFloat(track.dataset.percentage || "0");
      const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
      const nextPct = currentPct + (delta / (window.innerWidth / 2)) * -100;

      setPosition(nextPct);
      track.dataset.prevPercentage = track.dataset.percentage || "0";
    },
    { passive: false }
  );

  track.addEventListener("click", e => {
    if (moved) {
      e.preventDefault();
    }
  });

  track.addEventListener("dragstart", e => e.preventDefault());
});
});