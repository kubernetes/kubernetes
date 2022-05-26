# goconst

Find repeated strings that could be replaced by a constant.

### Motivation

There are obvious benefits to using constants instead of repeating strings, mostly to ease maintenance. Cannot argue against changing a single constant versus many strings.

While this could be considered a beginner mistake, across time, multiple packages and large codebases, some repetition could have slipped in.

### Get Started

    $ go get github.com/jgautheron/goconst/cmd/goconst
    $ goconst ./...

### Usage

```
Usage:

  goconst ARGS <directory>

Flags:

  -ignore            exclude files matching the given regular expression
  -ignore-tests      exclude tests from the search (default: true)
  -min-occurrences   report from how many occurrences (default: 2)
  -min-length        only report strings with the minimum given length (default: 3)
  -match-constant    look for existing constants matching the values
  -numbers           search also for duplicated numbers
  -min          	   minimum value, only works with -numbers
  -max          	   maximum value, only works with -numbers
  -output            output formatting (text or json)
  -set-exit-status   Set exit status to 2 if any issues are found

Examples:

  goconst ./...
  goconst -ignore "yacc|\.pb\." $GOPATH/src/github.com/cockroachdb/cockroach/...
  goconst -min-occurrences 3 -output json $GOPATH/src/github.com/cockroachdb/cockroach
  goconst -numbers -min 60 -max 512 .
```

### Other static analysis tools

- [gogetimports](https://github.com/jgautheron/gogetimports): Get a JSON-formatted list of imports.
- [usedexports](https://github.com/jgautheron/usedexports): Find exported variables that could be unexported.

### License
MIT
