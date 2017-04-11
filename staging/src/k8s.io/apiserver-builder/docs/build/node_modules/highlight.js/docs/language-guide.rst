Language definition guide
=========================

Highlighting overview
---------------------

Programming language code consists of parts with different rules of parsing: keywords like ``for`` or ``if``
don't make sense inside strings, strings may contain backslash-escaped symbols like ``\"``
and comments usually don't contain anything interesting except the end of the comment.

In highlight.js such parts are called "modes".

Each mode consists of:

* starting condition
* ending condition
* list of contained sub-modes
* lexing rules and keywords
* …exotic stuff like another language inside a language

The parser's work is to look for modes and their keywords.
Upon finding, it wraps them into the markup ``<span class="...">...</span>``
and puts the name of the mode ("string", "comment", "number")
or a keyword group name ("keyword", "literal", "built-in") as the span's class name.


General syntax
--------------

A language definition is a JavaScript object describing the default parsing mode for the language.
This default mode contains sub-modes which in turn contain other sub-modes, effectively making the language definition a tree of modes.

Here's an example:

::

  {
    case_insensitive: true, // language is case-insensitive
    keywords: 'for if while',
    contains: [
      {
        className: 'string',
        begin: '"', end: '"'
      },
      hljs.COMMENT(
        '/\\*', // begin
        '\\*/', // end
        {
          contains: [
            {
              className: 'doc', begin: '@\\w+'
            }
          ]
        }
      )
    ]
  }

Usually the default mode accounts for the majority of the code and describes all language keywords.
A notable exception here is XML in which a default mode is just a user text that doesn't contain any keywords,
and most interesting parsing happens inside tags.


Keywords
--------

In the simple case language keywords are defined in a string, separated by space:

::

  {
    keywords: 'else for if while'
  }

Some languages have different kinds of "keywords" that might not be called as such by the language spec
but are very close to them from the point of view of a syntax highlighter. These are all sorts of "literals", "built-ins", "symbols" and such.
To define such keyword groups the attribute ``keywords`` becomes an object each property of which defines its own group of keywords:

::

  {
    keywords: {
      keyword: 'else for if while',
      literal: 'false true null'
    }
  }

The group name becomes then a class name in a generated markup enabling different styling for different kinds of keywords.

To detect keywords highlight.js breaks the processed chunk of code into separate words — a process called lexing.
The "word" here is defined by the regexp ``[a-zA-Z][a-zA-Z0-9_]*`` that works for keywords in most languages.
Different lexing rules can be defined by the ``lexemes`` attribute:

::

  {
    lexemes '-[a-z]+',
    keywords: '-import -export'
  }


Sub-modes
---------

Sub-modes are listed in the ``contains`` attribute:

::

  {
    keywords: '...',
    contains: [
      hljs.QUOTE_STRING_MODE,
      hljs.C_LINE_COMMENT,
      { ... custom mode definition ... }
    ]
  }

A mode can reference itself in the ``contains`` array by using a special keyword ``'self``'.
This is commonly used to define nested modes:

::

  {
    className: 'object',
    begin: '{', end: '}',
    contains: [hljs.QUOTE_STRING_MODE, 'self']
  }


Comments
--------

To define custom comments it is recommended to use a built-in helper function ``hljs.COMMENT`` instead of describing the mode directly, as it also defines a few default sub-modes that improve language detection and do other nice things.

Parameters for the function are:

::

  hljs.COMMENT(
    begin,      // begin regex
    end,        // end regex
    extra       // optional object with extra attributes to override defaults
                // (for example {relevance: 0})
  )


Markup generation
-----------------

Modes usually generate actual highlighting markup — ``<span>`` elements with specific class names that are defined by the ``className`` attribute:

::

  {
    contains: [
      {
        className: 'string',
        // ... other attributes
      },
      {
        className: 'number',
        // ...
      }
    ]
  }

Names are not required to be unique, it's quite common to have several definitions with the same name.
For example, many languages have various syntaxes for strings, comments, etc…

Sometimes modes are defined only to support specific parsing rules and aren't needed in the final markup.
A classic example is an escaping sequence inside strings allowing them to contain an ending quote.

::

  {
    className: 'string',
    begin: '"', end: '"',
    contains: [{begin: '\\\\.'}],
  }

For such modes ``className`` attribute should be omitted so they won't generate excessive markup.


Mode attributes
---------------

Other useful attributes are defined in the :doc:`mode reference </reference>`.


.. _relevance:

Relevance
---------

Highlight.js tries to automatically detect the language of a code fragment.
The heuristics is essentially simple: it tries to highlight a fragment with all the language definitions
and the one that yields most specific modes and keywords wins. The job of a language definition
is to help this heuristics by hinting relative relevance (or irrelevance) of modes.

This is best illustrated by example. Python has special kinds of strings defined by prefix letters before the quotes:
``r"..."``, ``u"..."``. If a code fragment contains such strings there is a good chance that it's in Python.
So these string modes are given high relevance:

::

  {
    className: 'string',
    begin: 'r"', end: '"',
    relevance: 10
  }

On the other hand, conventional strings in plain single or double quotes aren't specific to any language
and it makes sense to bring their relevance to zero to lessen statistical noise:

::

  {
    className: 'string',
    begin: '"', end: '"',
    relevance: 0
  }

The default value for relevance is 1. When setting an explicit value it's recommended to use either 10 or 0.

Keywords also influence relevance. Each of them usually has a relevance of 1, but there are some unique names
that aren't likely to be found outside of their languages, even in the form of variable names.
For example just having ``reinterpret_cast`` somewhere in the code is a good indicator that we're looking at C++.
It's worth to set relevance of such keywords a bit higher. This is done with a pipe:

::

  {
    keywords: 'for if reinterpret_cast|10'
  }


Illegal symbols
---------------

Another way to improve language detection is to define illegal symbols for a mode.
For example in Python first line of class definition (``class MyClass(object):``) cannot contain symbol "{" or a newline.
Presence of these symbols clearly shows that the language is not Python and the parser can drop this attempt early.

Illegal symbols are defined as a a single regular expression:

::

  {
    className: 'class',
    illegal: '[${]'
  }


Pre-defined modes and regular expressions
-----------------------------------------

Many languages share common modes and regular expressions. Such expressions are defined in core highlight.js code
at the end under "Common regexps" and "Common modes" titles. Use them when possible.


Contributing
------------

Follow the :doc:`contributor checklist </language-contribution>`.
