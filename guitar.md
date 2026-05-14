---
layout: default
title: Guitar
permalink: /guitar/
---

<div class="home">

<h1 class="pageTitle">Guitar Practice</h1>

<div class="posts noList">

{% assign pieces = site.guitar | sort: "date" | reverse %}

{% for piece in pieces %}

<article>
  {% if piece.date %}
    <span class="date">{{ piece.date | date_to_string }}</span>
  {% endif %}

  <h3>
    <a href="{{ piece.url | relative_url }}">{{ piece.title }}</a>
  </h3>

  <p>
    {% if piece.capo %}Capo: {{ piece.capo }} | {% endif %}
    {% if piece.tempo %}Tempo: {{ piece.tempo }} BPM{% endif %}

    {% if piece.audio %}
      <audio class="inline-audio" controls preload="metadata">
        <source src="{{ site.baseurl }}/assets/audio/guitar/{{ piece.audio }}" type="audio/mp4">
        Your browser does not support the audio element.
      </audio>
    {% endif %}
  </p>
</article>

{% endfor %}

</div>
</div>