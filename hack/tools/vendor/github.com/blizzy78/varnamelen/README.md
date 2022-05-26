[![GoDoc](https://pkg.go.dev/badge/github.com/blizzy78/varnamelen)](https://pkg.go.dev/github.com/blizzy78/varnamelen)


varnamelen
==========

A Go Analyzer checking that the length of a variable's name matches its usage scope.

A variable with a short name can be hard to use if the variable is used over a longer span of lines of code.
A longer variable name may be easier to comprehend.

The analyzer can check variable names, method receiver names, as well as named return values.

Conventional Go parameters such as `ctx context.Context` or `t *testing.T` will always be ignored.

**Example output**

```
test.go:4:2: variable name 'x' is too short for the scope of its usage (varnamelen)
        x := 123
        ^
test.go:6:2: variable name 'i' is too short for the scope of its usage (varnamelen)
        i := 10
        ^
```


golangci-lint Integration
-------------------------

varnamelen is integrated into [golangci-lint] (though it may not always be the most recent version.)

Example configuration for golangci-lint:

```yaml
linters-settings:
  varnamelen:
    # The longest distance, in source lines, that is being considered a "small scope." (defaults to 5)
    # Variables used in at most this many lines will be ignored.
    max-distance: 5
    # The minimum length of a variable's name that is considered "long." (defaults to 3)
    # Variable names that are at least this long will be ignored.
    min-name-length: 3
    # Check method receiver names. (defaults to false)
    check-receiver: false
    # Check named return values. (defaults to false)
    check-return: false
    # Ignore "ok" variables that hold the bool return value of a type assertion. (defaults to false)
    ignore-type-assert-ok: false
    # Ignore "ok" variables that hold the bool return value of a map index. (defaults to false)
    ignore-map-index-ok: false
    # Ignore "ok" variables that hold the bool return value of a channel receive. (defaults to false)
    ignore-chan-recv-ok: false
    # Optional list of variable names that should be ignored completely. (defaults to empty list)
    ignore-names:
      - err
    # Optional list of variable declarations that should be ignored completely. (defaults to empty list)
    # Entries must be in the form of "<variable name> <type>" or "<variable name> *<type>" for
    # variables, or "const <name>" for constants.
    ignore-decls:
      - c echo.Context
      - t testing.T
      - f *foo.Bar
      - e error
      - i int
      - const C
```


Standalone Usage
----------------

The `cmd/` folder provides a standalone command line utility. You can build it like this:

```
go build -o varnamelen ./cmd/
```

**Usage**

```
varnamelen: checks that the length of a variable's name matches its scope

Usage: varnamelen [-flag] [package]

A variable with a short name can be hard to use if the variable is used
over a longer span of lines of code. A longer variable name may be easier
to comprehend.

Flags:
  -V	print version and exit
  -all
    	no effect (deprecated)
  -c int
    	display offending line with this many lines of context (default -1)
  -checkReceiver
    	check method receiver names
  -checkReturn
    	check named return values
  -cpuprofile string
    	write CPU profile to this file
  -debug string
    	debug flags, any subset of "fpstv"
  -fix
    	apply all suggested fixes
  -flags
    	print analyzer flags in JSON
  -ignoreChanRecvOk
    	ignore 'ok' variables that hold the bool return value of a channel receive
  -ignoreDecls value
    	comma-separated list of ignored variable declarations
  -ignoreMapIndexOk
    	ignore 'ok' variables that hold the bool return value of a map index
  -ignoreNames value
    	comma-separated list of ignored variable names
  -ignoreTypeAssertOk
    	ignore 'ok' variables that hold the bool return value of a type assertion
  -json
    	emit JSON output
  -maxDistance int
    	maximum number of lines of variable usage scope considered 'short' (default 5)
  -memprofile string
    	write memory profile to this file
  -minNameLength int
    	minimum length of variable name considered 'long' (default 3)
  -source
    	no effect (deprecated)
  -tags string
    	no effect (deprecated)
  -trace string
    	write trace log to this file
  -v	no effect (deprecated)
```


License
-------

This package is licensed under the MIT license.



[golangci-lint]: https://github.com/golangci/golangci-lint
