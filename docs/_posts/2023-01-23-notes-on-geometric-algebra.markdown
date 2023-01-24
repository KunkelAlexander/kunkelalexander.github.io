---
layout: post
title:  "Notes on Geometric Algebra!"
date:   2023-01-24 14:13:59 +0800
categories: jekyll update
---
In this post, we will take a close look at some concepts in geometric algebra and use them to build a python code that generates `n-dimensional rotations`. 

| Vector-Bivector | Vector-Vector |
|---|---|
|$$ a B = a\cdot B + a \wedge B $$|$$ u v = u\cdot v + u \wedge v $$|
|$$ a \cdot B = \frac{1}{2} \left(a B - B a\right) = \langle a B \rangle_1 $$|$$ u \cdot v = \frac{1}{2} \left(u v - v u\right) = \langle u v \rangle_0 $$|
|$$ a \wedge B = \frac{1}{2} \left(a B + B a\right) = \langle a B \rangle_3 $$|$$ u \wedge v = \frac{1}{2} \left(u v + v u\right) = \langle u v \rangle_2 $$|


{% highlight python %}
def myfunction(x):
	return x
{% endhighlight %}

Check out my [github][github] repository for the full code.

[github]: https://github.com/KunkelAlexander/
