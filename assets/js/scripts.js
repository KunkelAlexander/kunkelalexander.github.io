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
	let lastX = 0, velocity = 0;   // NEW

	const move = pct => {
	const trackW = track.scrollWidth;                          // NEW (dynamic clamp)
	const wrapW = wrap.clientWidth;                            // NEW
	const minPct = -((trackW - wrapW) / trackW) * 100;        // NEW
	pct = Math.max(Math.min(pct, 0), minPct);                 // changed from -90
	track.dataset.percentage = pct;
	track.animate(
		{ transform: `translate(${pct}%, -50%)` },
		{ duration: 900, fill: "forwards" }
	);
	for (const img of track.getElementsByClassName("image")) {
		img.animate(
		{ objectPosition: `${100 + pct}% center` },
		{ duration: 900, fill: "forwards" }
		);
	}
	};

	track.addEventListener("pointerdown", e => {
	moved = false;
	velocity = 0;                                              // NEW (reset on new drag)
	track.dataset.mouseDownAt = e.clientX;
	lastX = e.clientX;                                        // NEW
	track.setPointerCapture(e.pointerId);
	});

	track.addEventListener("pointerup", e => {
	track.dataset.mouseDownAt = "0";
	track.dataset.prevPercentage = track.dataset.percentage || "0";
	if (track.hasPointerCapture(e.pointerId)) track.releasePointerCapture(e.pointerId);

	// NEW — momentum coast
	let momentum = velocity * 1.5;
	const coast = () => {
		if (Math.abs(momentum) < 0.2) return;
		momentum *= 0.88;
		move(parseFloat(track.dataset.percentage || "0") + (momentum / window.innerWidth) * 100);
		track.dataset.prevPercentage = track.dataset.percentage;
		requestAnimationFrame(coast);
	};
	requestAnimationFrame(coast);
	});

	track.addEventListener("pointermove", e => {
	if (track.dataset.mouseDownAt === "0") return;
	velocity = e.clientX - lastX;                             // NEW
	lastX = e.clientX;                                        // NEW
	const delta = parseFloat(track.dataset.mouseDownAt) - e.clientX;
	if (Math.abs(delta) > 5) moved = true;
	const pct =
		parseFloat(track.dataset.prevPercentage || "0") +
		(delta / (window.innerWidth / 2)) * -100;
	move(pct);
	});

	track.addEventListener("click", e => {
	if (moved) e.preventDefault();
	});

	track.addEventListener("wheel", e => {
	e.preventDefault();
	const pct =
		parseFloat(track.dataset.percentage || "0") +
		(e.deltaY / (window.innerWidth / 2)) * -100;
	track.dataset.prevPercentage = pct;
	move(pct);
	}, { passive: false });

	track.addEventListener("dragstart", e => e.preventDefault());
});

});