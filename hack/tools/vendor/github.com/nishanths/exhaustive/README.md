## exhaustive [![Godoc][2]][1]

Check exhaustiveness of enum switch statements in Go source code.

```
go install github.com/nishanths/exhaustive/cmd/exhaustive@latest
```

For docs on the flags, the definition of enum, and the definition of
exhaustiveness, see [godocs.io][4].

For the changelog, see [CHANGELOG][changelog] in the wiki.

The package provides an `Analyzer` that follows the guidelines in the
[`go/analysis`][3] package; this should make it possible to integrate
exhaustive with your own analysis driver program.

## Example

Given the enum

```go
package token

type Token int

const (
	Add Token = iota
	Subtract
	Multiply
	Quotient
	Remainder
)
```

and the switch statement

```go
package calc

import "token"

func f(t token.Token) {
	switch t {
	case token.Add:
	case token.Subtract:
	case token.Multiply:
	default:
	}
}
```

running exhaustive will print

```
calc.go:6:2: missing cases in switch of type token.Token: Quotient, Remainder
```

## Contributing

Issues and pull requests are welcome. Before making a substantial
change, please discuss it in an issue.

[1]: https://godocs.io/github.com/nishanths/exhaustive
[2]: https://godocs.io/github.com/nishanths/exhaustive?status.svg
[3]: https://pkg.go.dev/golang.org/x/tools/go/analysis
[4]: https://godocs.io/github.com/nishanths/exhaustive
[changelog]: https://github.com/nishanths/exhaustive/wiki/CHANGELOG
