# Humane Units

Just a few functions for helping humanize times and sizes.

`go get` it as `github.com/dustin/go-humanize`, import it as
`"github.com/dustin/go-humanize"`, use it as `humanize`

## Sizes

This lets you take numbers like `82854982` and convert them to useful
strings like, `83MB` or `79MiB` (whichever you prefer).

Example:

    fmt.Printf("That file is %s.", humanize.Bytes(82854982))

## Times

This lets you take a `time.Time` and spit it out in relative terms.
For example, `12 seconds ago` or `3 days from now`.

Example:

    fmt.Printf("This was touched %s", humanize.Time(someTimeInstance))

Thanks to Kyle Lemons for the time implementation from an IRC
conversation one day.  It's pretty neat.

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

    fmt.Printf("You're my %s best friend.", humanize.Ordinal(193))

## Commas

Want to shove commas into numbers?  Be my guest.

    0 -> 0
    100 -> 100
    1000 -> 1,000
    1000000000 -> 1,000,000,000
    -100000 -> -100,000

Example:

    fmt.Printf("You owe $%s.\n", humanize.Comma(6582491))

## Ftoa

Nicer float64 formatter that removes trailing zeros.

    fmt.Printf("%f", 2.24)                   // 2.240000
    fmt.Printf("%s", humanize.Ftoa(2.24))    // 2.24
    fmt.Printf("%f", 2.0)                    // 2.000000
    fmt.Printf("%s", humanize.Ftoa(2.0))     // 2

## SI notation

Format numbers with [SI notation][sinotation].

Example:

    humanize.SI(0.00000000223, "M")    // 2.23nM


[odisc]: https://groups.google.com/d/topic/golang-nuts/l8NhI74jl-4/discussion
[sinotation]: http://en.wikipedia.org/wiki/Metric_prefix
