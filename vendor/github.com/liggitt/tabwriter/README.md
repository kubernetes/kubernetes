This repo is a drop-in replacement for the golang [text/tabwriter](https://golang.org/pkg/text/tabwriter/) package.

It is based on that package at [cf2c2ea8](https://github.com/golang/go/tree/cf2c2ea89d09d486bb018b1817c5874388038c3a/src/text/tabwriter) and inherits its license.

The following additional features are supported:
* `RememberWidths` flag allows remembering maximum widths seen per column even after Flush() is called.
* `RememberedWidths() []int` and `SetRememberedWidths([]int) *Writer` allows obtaining and transferring remembered column width between writers.
