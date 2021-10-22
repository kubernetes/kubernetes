# gnostic-summary

This directory contains a `gnostic` plugin that summarizes the contents of an
OpenAPI description.

    gnostic bookstore.json --summary-out=-

Here the `-` in the output path indicates that results are to be written to
stdout. A `.` will write a summary file into the current directory.
