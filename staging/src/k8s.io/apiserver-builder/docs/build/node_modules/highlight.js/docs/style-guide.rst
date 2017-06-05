Style guide
===========


Key principle
-------------

Highlight.js themes are language agnostic.

Instead of trying to make a *rich* set of highlightable classes look good in a
handful of languages we have a *limited* set of classes that work for all
languages.

Hence, there are two important implications:

* Highlight.js styles tend to be minimalistic.
* It's not possible to exactly emulate themes from other highlighting engines.


Defining a theme
----------------

A theme is a single CSS defining styles for class names listed in the
:doc:`class reference </css-classes-reference>`. The general guideline is to
style all available classes, however an author may deliberately choose to
exclude some (for example, ``.attr`` is usually left unstyled).

You are not required to invent a separate styling for every group of class
names, it's perfectly okay to group them:

::

  .hljs-string,
  .hljs-section,
  .hljs-selector-class,
  .hljs-template-variable,
  .hljs-deletion {
    color: #800;
  }

Use as few or as many unique style combinations as you want.


Typography and layout dos and don'ts
------------------------------------

Don't use:

* non-standard borders/margin/paddings for the root container ``.hljs``
* specific font faces
* font size, line height and anything that affects position and size of
  characters within the container

Okay to use:

* colors (obviously!)
* italic, bold, underlining, etc.
* image backgrounds

These may seem arbitrary at first but it's what has shown to make sense in
practice.

There's also a common set of rules that *has* to be defined for the root
container verbatim:

::

  .hljs {
    display: block;
    overflow-x: auto;
    padding: 0.5em;
  }


``.subst``
----------

One important caveat: don't forget to style ``.subst``. It's used for parsed
sections within strings and almost always should be reset to the default color:

::

  .hljs,
  .hljs-subst {
    color: black;
  }


Contributing
------------

You should include a comment at the top of the CSS file with attribution and
other meta data if necessary. The format is free:

::

  /*

  Fancy style (c) John Smith <email@domain.com>

  */

If you're a new contributor add yourself to the authors list in AUTHORS.*.txt
(use either English and/or Russian version). Also update CHANGES.md with your
contribution.

Send your contribution as a pull request on GitHub.
