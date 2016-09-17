# Consul `types` Package

The Go language has a strong type system built into the language.  The
`types` package corrals named types into a single package that is terminal in
`go`'s import graph.  The `types` package should not have any downstream
dependencies.  Each subsystem that defines its own set of types exists in its
own file, but all types are defined in the same package.

# Why

> Everything should be made as simple as possible, but not simpler.

`string` is a useful container and underlying type for identifiers, however
the `string` type is effectively opaque to the compiler in terms of how a
given string is intended to be used.  For instance, there is nothing
preventing the following from happening:

```go
// `map` of Widgets, looked up by ID
var widgetLookup map[string]*Widget
// ...
var widgetID string = "widgetID"
w, found := widgetLookup[widgetID]

// Bad!
var widgetName string = "name of widget"
w, found := widgetLookup[widgetName]
```

but this class of problem is entirely preventable:

```go
type WidgetID string
var widgetLookup map[WidgetID]*Widget
var widgetName
```

TL;DR: intentions and idioms aren't statically checked by compilers.  The
`types` package uses Go's strong type system to prevent this class of bug.
