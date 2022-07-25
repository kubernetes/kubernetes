# Slim-Sprig: Template functions for Go templates [![GoDoc](https://godoc.org/github.com/go-task/slim-sprig?status.svg)](https://godoc.org/github.com/go-task/slim-sprig) [![Go Report Card](https://goreportcard.com/badge/github.com/go-task/slim-sprig)](https://goreportcard.com/report/github.com/go-task/slim-sprig)

Slim-Sprig is a fork of [Sprig](https://github.com/Masterminds/sprig), but with
all functions that depend on external (non standard library) or crypto packages
removed.
The reason for this is to make this library more lightweight. Most of these
functions (specially crypto ones) are not needed on most apps, but costs a lot
in terms of binary size and compilation time.

## Usage

**Template developers**: Please use Slim-Sprig's [function documentation](https://go-task.github.io/slim-sprig/) for
detailed instructions and code snippets for the >100 template functions available.

**Go developers**: If you'd like to include Slim-Sprig as a library in your program,
our API documentation is available [at GoDoc.org](http://godoc.org/github.com/go-task/slim-sprig).

For standard usage, read on.

### Load the Slim-Sprig library

To load the Slim-Sprig `FuncMap`:

```go

import (
  "html/template"

  "github.com/go-task/slim-sprig"
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
