# gnostic-complexity

This directory contains a `gnostic` plugin that computes simple complexity
metrics of an API description.

    gnostic bookstore.json --complexity-out=.

Here the `.` in the output path indicates that results are to be written to the
current directory.

The complexity metrics are described in `metrics/complexity.proto`.
