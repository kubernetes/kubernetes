# Prettybench

A tool for transforming `go test`'s benchmark output a bit to make it nicer for humans.

## Problem

Go benchmarks are great, particularly when used in concert with benchcmp. But the output can be a bit hard to
read:

![before](/screenshots/before.png)

## Solution

    $ go get github.com/cespare/prettybench
    $ go test -bench=. | prettybench

![after](/screenshots/after.png)

* Column headers
* Columns are aligned
* Time output is adjusted to convenient units

## Notes

* Right now the units for the time are chosen based on the smallest value in the column.
* Prettybench has to buffer all the rows of output before it can print them (for column formatting), so you
  won't see intermediate progress. If you want to see that too, you could tee your output so that you see the
  unmodified version as well. If you do this, you'll want to use the prettybench's `-no-passthrough` flag so
  it doesn't print all the other lines (because then they'd be printed twice):

        $ go test -bench=. | tee >(prettybench -no-passthrough)

## To Do (maybe)

* Handle benchcmp output
* Change the units for non-time columns as well (these are generally OK though).
