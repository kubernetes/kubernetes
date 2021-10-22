# Embedded Interfaces in aux_files

Embedded interfaces in `aux_files` generate `unknown embedded interface XXX` errors.
See below for example of the problem:

```go
// source
import (
    alias "some.org/package/imported"
)

type Source interface {
    alias.Foreign
}
```

```go
// some.org/package/imported
type Foreign interface {
    Embedded
}

type Embedded interface {}
```

Attempting to generate a mock will result in an `unknown embedded interface Embedded`.
The issue is that the `fileParser` stores `auxInterfaces` underneath the package name
explicitly specified in the `aux_files` flag.

In the `parseInterface` method, there is an incorrect assumption about an embedded interface
always being in the source file.

```go
case *ast.Ident:
        // Embedded interface in this package.
        ei := p.auxInterfaces[""][v.String()]
        if ei == nil {
                return nil, p.errorf(v.Pos(), "unknown embedded interface %s", v.String())
        }
```
