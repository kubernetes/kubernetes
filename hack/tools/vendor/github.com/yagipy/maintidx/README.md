# maintidx
`maintidx` measures the maintainability index of each function.  
https://docs.microsoft.com/en-us/visualstudio/code-quality/code-metrics-maintainability-index-range-and-meaning

## Installation
### Go version < 1.16
```shell
go get -u github.com/yagipy/maintidx/cmd/maintidx
```

### Go version 1.16+
```shell
go install github.com/yagipy/maintidx/cmd/maintidx
```

## Usage
### standalone
```shell
maintidx ./...
```

### with go run
No installation required
```shell
go run github.com/yagipy/maintidx/cmd/maintidx ./...
```

### with go vet
```shell
go vet -vettool=`which maintidx` ./...
```

## Flag
```shell
Flags:
  -under int
    	show functions with maintainability index < N only. (default 20)
```

## TODO
- [ ] Setup execute env on container
- [ ] Impl cyc.Cyc.Calc()
- [ ] Move maintidx.Visitor.PrintHalstVol to halstval package
- [ ] Consider the necessity of halstvol.incrIfAllTrue
- [ ] Test under pkg file
