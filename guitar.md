---
layout: default
title: Guitar
permalink: /guitar/
---

<div class="home">

<h1 class="pageTitle">Guitar Practice</h1>

<div class="posts noList">

{% for piece in site.guitar %}

<article>

<h3>
<a href="{{ piece.url }}">{{ piece.title }}</a>
</h3>

<p>
Capo: {{ piece.capo }} | Tempo: {{ piece.tempo }} BPM
</p>

<p>{{ piece.excerpt | strip_html }}</p>

</article>

{% endfor %}

</div>
</div>