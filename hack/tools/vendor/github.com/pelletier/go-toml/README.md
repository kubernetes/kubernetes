# go-toml

Go library for the [TOML](https://toml.io/) format.

This library supports TOML version
[v1.0.0-rc.3](https://toml.io/en/v1.0.0-rc.3)

[![Go Reference](https://pkg.go.dev/badge/github.com/pelletier/go-toml.svg)](https://pkg.go.dev/github.com/pelletier/go-toml)
[![license](https://img.shields.io/github/license/pelletier/go-toml.svg)](https://github.com/pelletier/go-toml/blob/master/LICENSE)
[![Build Status](https://dev.azure.com/pelletierthomas/go-toml-ci/_apis/build/status/pelletier.go-toml?branchName=master)](https://dev.azure.com/pelletierthomas/go-toml-ci/_build/latest?definitionId=1&branchName=master)
[![codecov](https://codecov.io/gh/pelletier/go-toml/branch/master/graph/badge.svg)](https://codecov.io/gh/pelletier/go-toml)
[![Go Report Card](https://goreportcard.com/badge/github.com/pelletier/go-toml)](https://goreportcard.com/report/github.com/pelletier/go-toml)
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fpelletier%2Fgo-toml.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2Fpelletier%2Fgo-toml?ref=badge_shield)


## Development status

**ℹ️ Consider go-toml v2!**

The next version of go-toml is in [active development][v2-dev], and
[nearing completion][v2-map].

Though technically in beta, v2 is already more tested, [fixes bugs][v1-bugs],
and [much faster][v2-bench]. If you only need reading and writing TOML documents
(majority of cases), those features are implemented and the API unlikely to
change.

The remaining features (Document structure editing and tooling) will be added
shortly. While pull-requests are welcome on v1, no active development is
expected on it. When v2.0.0 is released, v1 will be deprecated.

👉 [go-toml v2][v2]

[v2]: https://github.com/pelletier/go-toml/tree/v2
[v2-map]: https://github.com/pelletier/go-toml/discussions/506
[v2-dev]: https://github.com/pelletier/go-toml/tree/v2
[v1-bugs]: https://github.com/pelletier/go-toml/issues?q=is%3Aissue+is%3Aopen+label%3Av2-fixed
[v2-bench]: https://github.com/pelletier/go-toml/tree/v2#benchmarks

## Features

Go-toml provides the following features for using data parsed from TOML documents:

* Load TOML documents from files and string data
* Easily navigate TOML structure using Tree
* Marshaling and unmarshaling to and from data structures
* Line & column position data for all parsed elements
* [Query support similar to JSON-Path](query/)
* Syntax errors contain line and column numbers

## Import

```go
import "github.com/pelletier/go-toml"
```

## Usage example

Read a TOML document:

```go
config, _ := toml.Load(`
[postgres]
user = "pelletier"
password = "mypassword"`)
// retrieve data directly
user := config.Get("postgres.user").(string)

// or using an intermediate object
postgresConfig := config.Get("postgres").(*toml.Tree)
password := postgresConfig.Get("password").(string)
```

Or use Unmarshal:

```go
type Postgres struct {
    User     string
    Password string
}
type Config struct {
    Postgres Postgres
}

doc := []byte(`
[Postgres]
User = "pelletier"
Password = "mypassword"`)

config := Config{}
toml.Unmarshal(doc, &config)
fmt.Println("user=", config.Postgres.User)
```

Or use a query:

```go
// use a query to gather elements without walking the tree
q, _ := query.Compile("$..[user,password]")
results := q.Execute(config)
for ii, item := range results.Values() {
    fmt.Printf("Query result %d: %v\n", ii, item)
}
```

## Documentation

The documentation and additional examples are available at
[pkg.go.dev](https://pkg.go.dev/github.com/pelletier/go-toml).

## Tools

Go-toml provides three handy command line tools:

* `tomll`: Reads TOML files and lints them.

    ```
    go install github.com/pelletier/go-toml/cmd/tomll
    tomll --help
    ```
* `tomljson`: Reads a TOML file and outputs its JSON representation.

    ```
    go install github.com/pelletier/go-toml/cmd/tomljson
    tomljson --help
    ```

 * `jsontoml`: Reads a JSON file and outputs a TOML representation.

    ```
    go install github.com/pelletier/go-toml/cmd/jsontoml
    jsontoml --help
    ```

### Docker image

Those tools are also available as a Docker image from
[dockerhub](https://hub.docker.com/r/pelletier/go-toml). For example, to
use `tomljson`:

```
docker run -v $PWD:/workdir pelletier/go-toml tomljson /workdir/example.toml
```

Only master (`latest`) and tagged versions are published to dockerhub. You
can build your own image as usual:

```
docker build -t go-toml .
```

## Contribute

Feel free to report bugs and patches using GitHub's pull requests system on
[pelletier/go-toml](https://github.com/pelletier/go-toml). Any feedback would be
much appreciated!

### Run tests

`go test ./...`

### Fuzzing

The script `./fuzz.sh` is available to
run [go-fuzz](https://github.com/dvyukov/go-fuzz) on go-toml.

## Versioning

Go-toml follows [Semantic Versioning](http://semver.org/). The supported version
of [TOML](https://github.com/toml-lang/toml) is indicated at the beginning of
this document. The last two major versions of Go are supported
(see [Go Release Policy](https://golang.org/doc/devel/release.html#policy)).

## License

The MIT License (MIT) + Apache 2.0. Read [LICENSE](LICENSE).
