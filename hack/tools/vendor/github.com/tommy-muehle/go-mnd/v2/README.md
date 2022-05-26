# go-mnd - Magic number detector for Golang

<img align="right" width="250px" src="https://github.com/tommy-muehle/go-mnd/blob/master/images/logo.png">

A vet analyzer to detect magic numbers.

> **What is a magic number?**  
> A magic number is a numeric literal that is not defined as a constant, but which may change, and therefore can be hard to update. It's considered a bad programming practice to use numbers directly in any source code without an explanation. It makes programs harder to read, understand, and maintain.

## Project status

![CI](https://github.com/tommy-muehle/go-mnd/workflows/CI/badge.svg)
[![Go Report Card](https://goreportcard.com/badge/github.com/tommy-muehle/go-mnd)](https://goreportcard.com/report/github.com/tommy-muehle/go-mnd)
[![codecov](https://codecov.io/gh/tommy-muehle/go-mnd/branch/master/graph/badge.svg)](https://codecov.io/gh/tommy-muehle/go-mnd)

## Install

### Local

This analyzer requires Golang in version >= 1.12 because it's depends on the **go/analysis** API.

```
go get -u github.com/tommy-muehle/go-mnd/v2/cmd/mnd
```

### Github action

You can run go-mnd as a GitHub action as follows:

```
name: Example workflow
on:
  push:
    branches:
      - master
  pull_request:
    branches:
      - master
jobs:
  tests:
    runs-on: ubuntu-latest
    env:
      GO111MODULE: on
    steps:
      - name: Checkout Source
        uses: actions/checkout@v2
      - name: Run go-mnd
        uses: tommy-muehle/go-mnd@master
        with:
          args: ./...
```

### GitLab CI

You can run go-mnd inside a GitLab CI pipeline as follows:

```
stages:
  - lint

go:lint:mnd:
  stage: lint
  needs: []
  image: golang:latest
  before_script:
    - go get -u github.com/tommy-muehle/go-mnd/cmd/mnd
    - go mod tidy
    - go mod vendor
  script:
    - go vet -vettool $(which mnd) ./...
```

### Homebrew

To install with [Homebrew](https://brew.sh/), run:

```
brew tap tommy-muehle/tap && brew install tommy-muehle/tap/mnd
```

### Docker

To get the latest available Docker image:

```
docker pull tommymuehle/go-mnd
```

### Windows

On Windows download the [latest release](https://github.com/tommy-muehle/go-mnd/releases).

## Usage

[![asciicast](https://asciinema.org/a/231021.svg)](https://asciinema.org/a/231021)

```
go vet -vettool $(which mnd) ./...
```

or directly

```
mnd ./...
```

or via Docker

```
docker run --rm -v "$PWD":/app -w /app tommymuehle/go-mnd:latest ./...
```

## Options

The ```-checks``` option let's you define a comma separated list of checks.

The ```-ignored-numbers``` option let's you define a comma separated list of numbers to ignore.  
For example: `-ignored-numbers=1000,10_000,3.14159264`

The ```-ignored-functions``` option let's you define a comma separated list of function name regexp patterns to exclude.  
For example: `-ignored-functions=math.*,http.StatusText,make`

The ```-ignored-files``` option let's you define a comma separated list of filename regexp patterns to exclude.  
For example: `-ignored-files=magic_.*.go,.*_numbers.go`

## Checks

By default this detector analyses arguments, assigns, cases, conditions, operations and return statements.

* argument

```
t := http.StatusText(200)
```

* assign

```
c := &http.Client{
    Timeout: 5 * time.Second,
}
```

* case

```
switch x {
    case 3:
}
```

* condition

```
if x > 7 {
}
```

* operation

```
var x, y int
y = 10 * x
```

* return

```
return 3
```

## Excludes

By default the numbers 0 and 1 as well as test files are excluded! 

### Further known excludes

The function "Date" in the "Time" package.

```
t := time.Date(2017, time.September, 26, 12, 13, 14, 0, time.UTC)
```

Additional custom excludes can be defined via option flag.

## Development

### Build

You can build the binary with:

```
make
```

### Tests

You can run all unit tests using:

```
make test
```

And with coverage report:

```
make test-coverage
```

### Docker image

You can also build locally the docker image by using the command:

```
make image
```

## Stickers

<p style="float: left;">
    <img alt="Stickers image" width="200px" src="https://github.com/tommy-muehle/go-mnd/blob/master/images/stickers.jpg" />
    <img alt="Sticker image" width="200px" src="https://github.com/tommy-muehle/go-mnd/blob/master/images/sticker.jpg" />
</p>

Just drop me a message via Twitter DM or email if you want some go-mnd stickers
for you or your Gopher usergroup.

## License

The MIT License (MIT). Please see [LICENSE](LICENSE) for more information.
