# Humane Units [![Build Status](https://travis-ci.org/dustin/go-humanize.svg?branch=master)](https://travis-ci.org/dustin/go-humanize) [![GoDoc](https://godoc.org/github.com/dustin/go-humanize?status.svg)](https://godoc.org/github.com/dustin/go-humanize)

Just a few functions for helping humanize times and sizes.

`go get` it as `github.com/dustin/go-humanize`, import it as
`"github.com/dustin/go-humanize"`, use it as `humanize`.

See [godoc](https://pkg.go.dev/github.com/dustin/go-humanize) for
complete documentation.

## Sizes

This lets you take numbers like `82854982` and convert them to useful
strings like, `83 MB` or `79 MiB` (whichever you prefer).

Example:

```go
fmt.Printf("That file is %s.", humanize.Bytes(82854982)) // That file is 83 MB.
```

## Times

This lets you take a `time.Time` and spit it out in relative terms.
For example, `12 seconds ago` or `3 days from now`.

Example:

```go
fmt.Printf("This was touched %s.", humanize.Time(someTimeInstance)) // This was touched 7 hours ago.
```

Thanks to Kyle Lemons for the time implementation from an IRC
conversation one day. It's pretty neat.

## Ordinals

From a [mailing list discussion][odisc] where a user wanted to be able
to label ordinals.

    0 -> 0th
    1 -> 1st
    2 -> 2nd
    3 -> 3rd
    4 -> 4th
    [...]

Example:

```go
fmt.Printf("You're my %s best friend.", humanize.Ordinal(193)) // You are my 193rd best friend.
```

## Commas

Want to shove commas into numbers? Be my guest.

    0 -> 0
    100 -> 100
    1000 -> 1,000
    1000000000 -> 1,000,000,000
    -100000 -> -100,000

Example:

```go
fmt.Printf("You owe $%s.\n", humanize.Comma(6582491)) // You owe $6,582,491.
```

## Ftoa

Nicer float64 formatter that removes trailing zeros.

```go
fmt.Printf("%f", 2.24)                // 2.240000
fmt.Printf("%s", humanize.Ftoa(2.24)) // 2.24
fmt.Printf("%f", 2.0)                 // 2.000000
fmt.Printf("%s", humanize.Ftoa(2.0))  // 2
```

## SI notation

Format numbers with [SI notation][sinotation].

Example:

```go
humanize.SI(0.00000000223, "M") // 2.23 nM
```

## English-specific functions

The following functions are in the `humanize/english` subpackage.

### Plurals

Simple English pluralization

```go
english.PluralWord(1, "object", "") // object
english.PluralWord(42, "object", "") // objects
english.PluralWord(2, "bus", "") // buses
english.PluralWord(99, "locus", "loci") // loci

english.Plural(1, "object", "") // 1 object
english.Plural(42, "object", "") // 42 objects
english.Plural(2, "bus", "") // 2 buses
english.Plural(99, "locus", "loci") // 99 loci
```

### Word series

Format comma-separated words lists with conjuctions:

```go
english.WordSeries([]string{"foo"}, "and") // foo
english.WordSeries([]string{"foo", "bar"}, "and") // foo and bar
english.WordSeries([]string{"foo", "bar", "baz"}, "and") // foo, bar and baz

english.OxfordWordSeries([]string{"foo", "bar", "baz"}, "and") // foo, bar, and baz
```

[odisc]: https://groups.google.com/d/topic/golang-nuts/l8NhI74jl-4/discussion
[sinotation]: http://en.wikipedia.org/wiki/Metric_prefix
