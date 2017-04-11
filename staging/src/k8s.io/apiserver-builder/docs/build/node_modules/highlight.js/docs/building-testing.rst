Building and testing
====================

To actually run highlight.js it is necessary to build it for the environment
where you're going to run it: a browser, the node.js server, etc.


Building
--------

The build tool is written in JavaScript using node.js. Before running the
script, make sure to have node installed and run ``npm install`` to get the
dependencies.

The tool is located in ``tools/build.js``. A few useful examples:

* Build for a browser using only common languages::

    node tools/build.js :common

* Build for node.js including all available languages::

    node tools/build.js -t node

* Build two specific languages for debugging, skipping compression in this case::

    node tools/build.js -n python ruby

On some systems the node binary is named ``nodejs``; simply replace ``node``
with ``nodejs`` in the examples above if that is the case.

The full option reference is available with the usual ``--help`` option.

The build result will be in the ``build/`` directory.

.. _basic-testing:

Basic testing
-------------

The usual approach to debugging and testing a language is first doing it
visually. You need to build highlight.js with only the language you're working
on (without compression, to have readable code in browser error messages) and
then use the Developer tool in ``tools/developer.html`` to see how it highlights
a test snippet in that language.

A test snippet should be short and give the idea of the overall look of the
language. It shouldn't include every possible syntactic element and shouldn't
even make practical sense.

After you satisfied with the result you need to make sure that language
detection still works with your language definition included in the whole suite.

Testing is done using `Mocha <http://mochajs.org/>`_ and the
files are found in the ``test/`` directory. You can use the node build to
run the tests in the command line with ``npm test`` after installing the
dependencies with ``npm install``.

**Note**: for Debian-based machine, like Ubuntu, you might need to create an
alias or symbolic link for nodejs to node. The reason for this is the
dependencies that are requires to test highlight.js has a reference to
"node".

Place the snippet you used inside the browser in
``test/detect/<language>/default.txt``, build the package with all the languages
for node and run the test suite. If your language breaks auto-detection, it
should be fixed by :ref:`improving relevance <relevance>`, which is a black art
in and of itself. When in doubt, please refer to the discussion group!


Testing markup
--------------

You can also provide additional markup tests for the language to test isolated
cases of various syntactic construct. If your language has 19 different string
literals or complicated heuristics for telling division (``/``) apart from
regexes (``/ .. /``) -- this is the place.

A test case consists of two files:

* ``test/markup/<language>/<test_name>.txt``: test code
* ``test/markup/<language>/<test_name>.expect.txt``: reference rendering

To generate reference rendering use the Developer tool located at
``tools/developer.html``. Make sure to explicitly select your language in the
drop-down menu, as automatic detection is unlikely to work in this case.


