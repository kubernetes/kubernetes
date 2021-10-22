# gnostic-analyze

This directory contains a `gnostic` plugin that analyzes an OpenAPI description
for factors that might influence code generation and other API automation.

The plugin can be invoked like this:

    gnostic bookstore.json --analyze_out=.

This will write analysis results to a file in the current directory. Results
are written to a file named `summary.json`.

The plugin can be applied to a directory of descriptions using a command like
the following:

    find APIs -name "swagger.yaml" -exec gnostic --analyze_out=analysis {} \;

This finds all `swagger.yaml` files in a directory named `APIs` and its
subdirectories and writes corresponding `summary.json` files into a directory
named `analysis`.

Results of multiple analysis runs can be gathered together and summarized using
the `summarize` program, which is in the `summarize` subdirectory. Just run
`summarize` in the same location as the `find` command shown above.
