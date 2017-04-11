# Markdown is broken

I have a lot of scraps of markdown engine oddities that I've collected over the
years. What you see below is slightly messy, but it's what I've managed to
cobble together to illustrate the differences between markdown engines, and
why, if there ever is a markdown specification, it has to be absolutely
thorough. There are a lot more of these little differences I have documented
elsewhere. I know I will find them lingering on my disk one day, but until
then, I'll continue to add whatever strange nonsensical things I find.

Some of these examples may only mention a particular engine compared to marked.
However, the examples with markdown.pl could easily be swapped out for
discount, upskirt, or markdown.js, and you would very easily see even more
inconsistencies.

A lot of this was written when I was very unsatisfied with the inconsistencies
between markdown engines. Please excuse the frustration noticeable in my
writing.

## Examples of markdown's "stupid" list parsing

```
$ markdown.pl

  * item1

    * item2

  text
^D
<ul>
<li><p>item1</p>

<ul>
<li>item2</li>
</ul>

<p><p>text</p></li>
</ul></p>
```


```
$ marked
  * item1

    * item2

  text
^D
<ul>
<li><p>item1</p>
<ul>
<li>item2</li>
</ul>
<p>text</p>
</li>
</ul>
```

Which looks correct to you?

- - -

```
$ markdown.pl
* hello
  > world
^D
<p><ul>
<li>hello</p>

<blockquote>
  <p>world</li>
</ul></p>
</blockquote>
```

```
$ marked
* hello
  > world
^D
<ul>
<li>hello<blockquote>
<p>world</p>
</blockquote>
</li>
</ul>
```

Again, which looks correct to you?

- - -

EXAMPLE:

```
$ markdown.pl
* hello
  * world
    * hi
          code
^D
<ul>
<li>hello
<ul>
<li>world</li>
<li>hi
  code</li>
</ul></li>
</ul>
```

The code isn't a code block even though it's after the bullet margin. I know,
lets give it two more spaces, effectively making it 8 spaces past the bullet.

```
$ markdown.pl
* hello
  * world
    * hi
            code
^D
<ul>
<li>hello
<ul>
<li>world</li>
<li>hi
    code</li>
</ul></li>
</ul>
```

And, it's still not a code block. Did you also notice that the 3rd item isn't
even its own list? Markdown screws that up too because of its indentation
unaware parsing.

- - -

Let's look at some more examples of markdown's list parsing:

```
$ markdown.pl

  * item1

    * item2

  text
^D
<ul>
<li><p>item1</p>

<ul>
<li>item2</li>
</ul>

<p><p>text</p></li>
</ul></p>
```

Misnested tags.


```
$ marked
  * item1

    * item2

  text
^D
<ul>
<li><p>item1</p>
<ul>
<li>item2</li>
</ul>
<p>text</p>
</li>
</ul>
```

Which looks correct to you?

- - -

```
$ markdown.pl
* hello
  > world
^D
<p><ul>
<li>hello</p>

<blockquote>
  <p>world</li>
</ul></p>
</blockquote>
```

More misnested tags.


```
$ marked
* hello
  > world
^D
<ul>
<li>hello<blockquote>
<p>world</p>
</blockquote>
</li>
</ul>
```

Again, which looks correct to you?

- - -

# Why quality matters - Part 2

``` bash
$ markdown.pl
* hello
  > world
^D
<p><ul>
<li>hello</p>

<blockquote>
  <p>world</li>
</ul></p>
</blockquote>
```

``` bash
$ sundown # upskirt
* hello
  > world
^D
<ul>
<li>hello
&gt; world</li>
</ul>
```

``` bash
$ marked
* hello
  > world
^D
<ul><li>hello <blockquote><p>world</p></blockquote></li></ul>
```

Which looks correct to you?

- - -

See: https://github.com/evilstreak/markdown-js/issues/23

``` bash
$ markdown.pl # upskirt/markdown.js/discount
* hello
      var a = 1;
* world
^D
<ul>
<li>hello
var a = 1;</li>
<li>world</li>
</ul>
```

``` bash
$ marked
* hello
      var a = 1;
* world
^D
<ul><li>hello
<pre>code>var a = 1;</code></pre></li>
<li>world</li></ul>
```

Which looks more reasonable? Why shouldn't code blocks be able to appear in
list items in a sane way?

- - -

``` bash
$ markdown.js
<div>hello</div>

<span>hello</span>
^D
<p>&lt;div&gt;hello&lt;/div&gt;</p>

<p>&lt;span&gt;hello&lt;/span&gt;</p>
```

``` bash
$ marked
<div>hello</div>

<span>hello</span>
^D
<div>hello</div>


<p><span>hello</span>
</p>
```

- - -

See: https://github.com/evilstreak/markdown-js/issues/27

``` bash
$ markdown.js
[![an image](/image)](/link)
^D
<p><a href="/image)](/link">![an image</a></p>
```

``` bash
$ marked
[![an image](/image)](/link)
^D
<p><a href="/link"><img src="/image" alt="an image"></a>
</p>
```

- - -

See: https://github.com/evilstreak/markdown-js/issues/24

``` bash
$ markdown.js
> a

> b

> c
^D
<blockquote><p>a</p><p>bundefined&gt; c</p></blockquote>
```

``` bash
$ marked
> a

> b

> c
^D
<blockquote><p>a

</p></blockquote>
<blockquote><p>b

</p></blockquote>
<blockquote><p>c
</p></blockquote>
```

- - -

``` bash
$ markdown.pl
* hello
  * world
    how

    are
    you

  * today
* hi
^D
<ul>
<li><p>hello</p>

<ul>
<li>world
how</li>
</ul>

<p>are
you</p>

<ul>
<li>today</li>
</ul></li>
<li>hi</li>
</ul>
```

``` bash
$ marked
* hello
  * world
    how

    are
    you

  * today
* hi
^D
<ul>
<li><p>hello</p>
<ul>
<li><p>world
how</p>
<p>are
you</p>
</li>
<li><p>today</p>
</li>
</ul>
</li>
<li>hi</li>
</ul>
```
