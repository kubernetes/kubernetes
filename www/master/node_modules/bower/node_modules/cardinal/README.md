# cardinal [![Build Status](https://secure.travis-ci.org/thlorenz/cardinal.png)](http://travis-ci.org/thlorenz/cardinal)

[![NPM](https://nodei.co/npm/cardinal.png?downloads=true&stars=true)](https://nodei.co/npm/cardinal/)

**car·di·nal** *(kärdn-l, kärdnl)* - crested thick-billed North American finch having bright red plumage in the male.

![screenshot](https://github.com/thlorenz/cardinal/raw/master/assets/screen-shot.png)

## Features

- highlights JavaScript code with ANSI colors to improve terminal output
- theming support, see [custom color themes](https://github.com/thlorenz/cardinal/tree/master/themes)
- optionally print line numbers
- API and command line interface (`cdl`)
- `.cardinalrc` config to customize settings
- supports UNIX pipes

***

**Table of Contents**  *generated with [DocToc](http://doctoc.herokuapp.com/)*

- [Installation](#installation)
  - [As library](#as-library)
  - [As Commandline Tool](#as-commandline-tool)
- [Commandline](#commandline)
  - [Highlight a file](#highlight-a-file)
  - [As part of a UNIX pipe](#as-part-of-a-unix-pipe)
  - [Theme](#theme)
- [API](#api)
  - [*highlight(code[, opts])*](#highlightcode-opts)
  - [*highlightFileSync(fullPath[, opts])*](#highlightfilesyncfullpath-opts)
  - [*highlightFile(fullPath[, opts], callback)*](#highlightfilefullpath-opts-callback)
  - [opts](#opts)
- [Examples ([*browse*](https://github.com/thlorenz/cardinal/tree/master/examples))](#examples-[browse]https://githubcom/thlorenz/cardinal/tree/master/examples)


## Installation

### As library

    npm install cardinal

### As Commandline Tool

    [sudo] npm install -g cardinal

**Note:** 

When installed globally, cardinal exposes itself as the `cdl` command.

## Commandline

### Highlight a file

    cdl <file.js> [options]

**options**:
  - `--nonum`: turns off line number printing (relevant if it is turned on inside `~/.cardinalrc`

### As part of a UNIX pipe

    cat file.js | grep console | cdl

**Note:**

Not all code lines may be parsable JavaScript. In these cases the line is printed to the terminal without
highlighting it.

### Theme

The default theme will be used for highlighting.

To use a different theme, include a `.cardinalrc` file in your `HOME` directory.

This is a JSON file of the following form:

```json
{
  "theme": "hide-semicolons",
  "linenos": true|false
}
```

- `theme` can be the name of any of the [built-in themes](https://github.com/thlorenz/cardinal/tree/master/themes) or the
full path to a custom theme anywhere on your computer.
- linenos toggles line number printing

## API

### *highlight(code[, opts])*

- returns the highlighted version of the passed code ({String}) or throws an error if it was not able to parse it
- opts (see below)

### *highlightFileSync(fullPath[, opts])*

- returns the highlighted version of the file whose fullPath ({String}) was passed or throws an error if it was not able
  to parse it
- opts (see below)

### *highlightFile(fullPath[, opts], callback)*

- calls back with the highlighted version of the file whose fullPath ({String}) was passed or with an error if it was not able
  to parse it
- opts (see below)
- `callback` ({Function}) has the following signature: `function (err, highlighted) { .. }`

### opts

opts is an {Object} with the following properties:

- `theme` {Object} is used to optionally override the theme used to highlight
- `linenos` {Boolean} if `true` line numbers are included in the highlighted code
- `firstline` {Integer} sets line number of the first line when line numbers are printed
- `json` {Boolean} if `true` highlights JSON in addition to JavaScript (`true` by default if file extension is `.json`)

## Examples ([*browse*](https://github.com/thlorenz/cardinal/tree/master/examples))

- [sample .cardinalrc](https://github.com/thlorenz/cardinal/blob/master/examples/.cardinalrc)
- [highlighting a code snippet](https://github.com/thlorenz/cardinal/blob/master/examples/highlight-string.js) via
  ***highlight()***
- [file that highlights itself](https://github.com/thlorenz/cardinal/blob/master/examples/highlight-self.js) via
  ***highlightFile()*** including line numbers
- [file that highlights itself hiding all
  semicolons](https://github.com/thlorenz/cardinal/blob/master/examples/highlight-self-hide-semicolons.js) via
  ***highlightFileSync()***




[![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/thlorenz/cardinal/trend.png)](https://bitdeli.com/free "Bitdeli Badge")

