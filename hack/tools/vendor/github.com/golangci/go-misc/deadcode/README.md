# deadcode

`deadcode` is a very simple utility which detects unused declarations in a Go package.

## Usage
```
deadcode [-test] [packages]

    -test     Include test files
    packages  A list of packages using the same conventions as the go tool
```

## Limitations

* Self-referential unused code is not currently reported
* A single package can be tested at a time
* Unused methods are not reported

