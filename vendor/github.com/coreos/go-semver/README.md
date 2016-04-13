# go-semver - Semantic Versioning Library

[![Build Status](https://travis-ci.org/coreos/go-semver.png)](https://travis-ci.org/coreos/go-semver)

go-semver is a [semantic versioning][semver] library for Go. It lets you parse
and compare two semantic version strings.

[semver]: http://semver.org/

## Usage

```
vA, err := semver.NewVersion("1.2.3")
vB, err := semver.NewVersion("3.2.1")

fmt.Printf("%s < %s == %t\n", vA, vB, vA.LessThan(*vB))
```

## Example Application

```
$ go run example.go 1.2.3 3.2.1
1.2.3 < 3.2.1 == true

$ go run example.go 5.2.3 3.2.1
5.2.3 < 3.2.1 == false
```

## TODO

- Richer comparision operations
