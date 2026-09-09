go.yaml.in/yaml
===============

YAML Support for the Go Language


## Introduction

The `yaml` package enables [Go](https://go.dev/) programs to comfortably encode
and decode [YAML](https://yaml.org/) values.

It was originally developed within [Canonical](https://www.canonical.com) as
part of the [juju](https://juju.ubuntu.com) project, and is based on a pure Go
port of the well-known [libyaml](http://pyyaml.org/wiki/LibYAML) C library to
parse and generate YAML data quickly and reliably.


## Project Status

This project started as a fork of the extremely popular [go-yaml](
https://github.com/go-yaml/yaml/)
project, and is being maintained by the official [YAML organization](
https://github.com/yaml/).

The YAML team took over ongoing maintenance and development of the project after
discussion with go-yaml's author, @niemeyer, following his decision to
[label the project repository as "unmaintained"](
https://github.com/go-yaml/yaml/blob/944c86a7d2/README.md) in April 2025.

We have put together a team of dedicated maintainers including representatives
of go-yaml's most important downstream projects.

We will strive to earn the trust of the various go-yaml forks to switch back to
this repository as their upstream.

Please [contact us](https://cloud-native.slack.com/archives/C08PPAT8PS7) if you
would like to contribute or be involved.


## Compatibility

The `yaml` package supports most of YAML 1.2, but preserves some behavior from
1.1 for backwards compatibility.

Specifically, v3 of the `yaml` package:

* Supports YAML 1.1 bools (`yes`/`no`, `on`/`off`) as long as they are being
  decoded into a typed bool value.
  Otherwise they behave as a string.
  Booleans in YAML 1.2 are `true`/`false` only.
* Supports octals encoded and decoded as `0777` per YAML 1.1, rather than
  `0o777` as specified in YAML 1.2, because most parsers still use the old
  format.
  Octals in the `0o777` format are supported though, so new files work.
* Does not support base-60 floats.
  These are gone from YAML 1.2, and were actually never supported by this
  package as it's clearly a poor choice.


## Installation and Usage

The import path for the package is *go.yaml.in/yaml/v3*.

To install it, run:

```bash
go get go.yaml.in/yaml/v3
```


## API Documentation

See: <https://pkg.go.dev/go.yaml.in/yaml/v3>


## API Stability

The package API for yaml v3 will remain stable as described in [gopkg.in](
https://gopkg.in).


## Example

```go
package main

import (
	"fmt"
	"log"

	"go.yaml.in/yaml/v3"
)

var data = `
a: Easy!
b:
  c: 2
  d: [3, 4]
`

// Note: struct fields must be public in order for unmarshal to
// correctly populate the data.
type T struct {
	A string
	B struct {
		RenamedC int   `yaml:"c"`
		D	[]int `yaml:",flow"`
	}
}

func main() {
	t := T{}

	err := yaml.Unmarshal([]byte(data), &t)
	if err != nil {
		log.Fatalf("error: %v", err)
	}
	fmt.Printf("--- t:\n%v\n\n", t)

	d, err := yaml.Marshal(&t)
	if err != nil {
		log.Fatalf("error: %v", err)
	}
	fmt.Printf("--- t dump:\n%s\n\n", string(d))

	m := make(map[interface{}]interface{})

	err = yaml.Unmarshal([]byte(data), &m)
	if err != nil {
		log.Fatalf("error: %v", err)
	}
	fmt.Printf("--- m:\n%v\n\n", m)

	d, err = yaml.Marshal(&m)
	if err != nil {
		log.Fatalf("error: %v", err)
	}
	fmt.Printf("--- m dump:\n%s\n\n", string(d))
}
```

This example will generate the following output:

```
--- t:
{Easy! {2 [3 4]}}

--- t dump:
a: Easy!
b:
  c: 2
  d: [3, 4]


--- m:
map[a:Easy! b:map[c:2 d:[3 4]]]

--- m dump:
a: Easy!
b:
  c: 2
  d:
  - 3
  - 4
```


## License

The yaml package is licensed under the MIT and Apache License 2.0 licenses.
Please see the LICENSE file for details.
