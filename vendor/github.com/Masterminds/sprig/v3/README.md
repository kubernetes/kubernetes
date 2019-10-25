# Sprig: Template functions for Go templates [![GoDoc](https://godoc.org/github.com/Masterminds/sprig?status.svg)](https://godoc.org/github.com/Masterminds/sprig) [![Go Report Card](https://goreportcard.com/badge/github.com/Masterminds/sprig)](https://goreportcard.com/report/github.com/Masterminds/sprig)

[![Stability: Sustained](https://masterminds.github.io/stability/sustained.svg)](https://masterminds.github.io/stability/sustained.html)
[![Build Status](https://travis-ci.org/Masterminds/sprig.svg?branch=master)](https://travis-ci.org/Masterminds/sprig)

The Go language comes with a [built-in template
language](http://golang.org/pkg/text/template/), but not
very many template functions. Sprig is a library that provides more than 100 commonly
used template functions.

It is inspired by the template functions found in
[Twig](http://twig.sensiolabs.org/documentation) and in various
JavaScript libraries, such as [underscore.js](http://underscorejs.org/).

## Package Versions

There are two active major versions of the `sprig` package.

* v3 is currently stable release series on the `master` branch. The Go API should
  remain compatible with v2, the current stable version. Behavior change behind
  some functions is the reason for the new major version.
* v2 is the previous stable release series. It has been more than three years since
  the initial release of v2. You can read the documentation and see the code
  on the [release-2](https://github.com/Masterminds/sprig/tree/release-2) branch.
  Bug fixes to this major version will continue for some time.

## Usage

**Template developers**: Please use Sprig's [function documentation](http://masterminds.github.io/sprig/) for
detailed instructions and code snippets for the >100 template functions available.

**Go developers**: If you'd like to include Sprig as a library in your program,
our API documentation is available [at GoDoc.org](http://godoc.org/github.com/Masterminds/sprig).

For standard usage, read on.

### Load the Sprig library

To load the Sprig `FuncMap`:

```go

import (
  "github.com/Masterminds/sprig"
  "html/template"
)

// This example illustrates that the FuncMap *must* be set before the
// templates themselves are loaded.
tpl := template.Must(
  template.New("base").Funcs(sprig.FuncMap()).ParseGlob("*.html")
)


```

### Calling the functions inside of templates

By convention, all functions are lowercase. This seems to follow the Go
idiom for template functions (as opposed to template methods, which are
TitleCase). For example, this:

```
{{ "hello!" | upper | repeat 5 }}
```

produces this:

```
HELLO!HELLO!HELLO!HELLO!HELLO!
```

## Principles Driving Our Function Selection

We followed these principles to decide which functions to add and how to implement them:

- Use template functions to build layout. The following
  types of operations are within the domain of template functions:
  - Formatting
  - Layout
  - Simple type conversions
  - Utilities that assist in handling common formatting and layout needs (e.g. arithmetic)
- Template functions should not return errors unless there is no way to print
  a sensible value. For example, converting a string to an integer should not
  produce an error if conversion fails. Instead, it should display a default
  value.
- Simple math is necessary for grid layouts, pagers, and so on. Complex math
  (anything other than arithmetic) should be done outside of templates.
- Template functions only deal with the data passed into them. They never retrieve
  data from a source.
- Finally, do not override core Go template functions.
