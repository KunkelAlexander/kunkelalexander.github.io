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

  const getBounds = () => {
    const cards = track.querySelectorAll(".project-card");
    if (!cards.length) {
      return { minPct: 0, maxPct: 0 };
    }

    const first = cards[0];
    const last = cards[cards.length - 1];

    const wrapCenter = wrap.clientWidth / 2;
    const trackWidth = track.scrollWidth;

    const firstCenter = first.offsetLeft + first.offsetWidth / 2;
    const lastCenter = last.offsetLeft + last.offsetWidth / 2;

    // Because .project-track is left: 50%, translate(0%) puts its left edge
    // at the center of the wrapper. These pixel offsets center first/last card.
    const maxPx = wrapCenter - firstCenter;
    const minPx = wrapCenter - lastCenter;

    return {
      maxPct: (maxPx / trackWidth) * 100,
      minPct: (minPx / trackWidth) * 100
    };
  };

  const clamp = pct => {
    const { minPct, maxPct } = getBounds();
    return Math.max(Math.min(pct, maxPct), minPct);
  };

  const move = pct => {
    const nextPct = clamp(pct);

    track.dataset.percentage = nextPct;

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
  };

  const centerFirstCard = () => {
    const { maxPct } = getBounds();
    track.dataset.prevPercentage = maxPct;
    track.dataset.percentage = maxPct;

    track.style.transform = `translate(${maxPct}%, -50%)`;

    for (const img of track.getElementsByClassName("image")) {
      img.style.objectPosition = `${100 + maxPct}% center`;
    }
  };

  requestAnimationFrame(centerFirstCard);
  window.addEventListener("resize", centerFirstCard);

  const handleDown = e => {
    moved = false;
    track.dataset.mouseDownAt = e.clientX;
  };

  const handleUp = () => {
    track.dataset.mouseDownAt = "0";
    track.dataset.prevPercentage = track.dataset.percentage || "0";
  };

  const handleMove = e => {
    if (track.dataset.mouseDownAt === "0") return;

    const mouseDelta = parseFloat(track.dataset.mouseDownAt) - e.clientX;
    const maxDelta = window.innerWidth / 2;

    if (Math.abs(mouseDelta) > 5) moved = true;

    const percentage = (mouseDelta / maxDelta) * -100;
    const nextPercentageUnconstrained =
      parseFloat(track.dataset.prevPercentage || "0") + percentage;

    move(nextPercentageUnconstrained);
  };

  window.addEventListener("mousedown", handleDown);
  window.addEventListener("mouseup", handleUp);
  window.addEventListener("mousemove", handleMove);

  window.addEventListener("touchstart", e => handleDown(e.touches[0]), {
    passive: true
  });

  window.addEventListener("touchend", handleUp);

  window.addEventListener("touchmove", e => handleMove(e.touches[0]), {
    passive: true
  });

  track.addEventListener("click", e => {
    if (moved) e.preventDefault();
  });

  track.addEventListener("dragstart", e => e.preventDefault());
});

});