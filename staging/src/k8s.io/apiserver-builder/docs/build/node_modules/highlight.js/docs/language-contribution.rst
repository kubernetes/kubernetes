Language contributor checklist
==============================

1. Put language definition into a .js file
------------------------------------------

The file defines a function accepting a reference to the library and returning a language object.
The library parameter is useful to access common modes and regexps. You should not immediately call this function,
this is done during the build process and details differ for different build targets.

::

  function(hljs) {
    return {
      keywords: 'foo bar',
      contains: [ ..., hljs.NUMBER_MODE, ... ]
    }
  }

The name of the file is used as a short language identifier and should be usable as a class name in HTML and CSS.


2. Provide meta data
--------------------

At the top of the file there is a specially formatted comment with meta data processed by a build system.
Meta data format is simply key-value pairs each occupying its own line:

::

  /*
  Language: Superlanguage
  Requires: java.js, sql.js
  Author: John Smith <email@domain.com>
  Contributors: Mike Johnson <...@...>, Matt Wilson <...@...>
  Description: Some cool language definition
  */

``Language`` — the only required header giving a human-readable language name.

``Requires`` — a list of other language files required for this language to work.
This make it possible to describe languages that extend definitions of other ones.
Required files aren't processed in any special way.
The build system just makes sure that they will be in the final package in
``LANGUAGES`` object.

The meaning of the other headers is pretty obvious.


3. Create a code example
------------------------

The code example is used both to test language detection and for the demo page
on https://highlightjs.org/. Put it in ``test/detect/<language>/default.txt``.

Take inspiration from other languages in ``test/detect/`` and read
:ref:`testing instructions <basic-testing>` for more details.


4. Write class reference
------------------------

Class reference lives in the :doc:`CSS classes reference </css-classes-reference>`..
Describe shortly names of all meaningful modes used in your language definition.


5. Add yourself to AUTHORS.*.txt and CHANGES.md
-----------------------------------------------

If you're a new contributor add yourself to the authors list. Feel free to use
either English and/or Russian version.
Also it will be good to update CHANGES.md.


6. Create a pull request
------------------------

Send your contribution as a pull request on GitHub.
