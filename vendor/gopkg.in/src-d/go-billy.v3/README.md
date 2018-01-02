# go-billy [![GoDoc](https://godoc.org/gopkg.in/src-d/go-billy.v3?status.svg)](https://godoc.org/gopkg.in/src-d/go-billy.v3) [![Build Status](https://travis-ci.org/src-d/go-billy.svg)](https://travis-ci.org/src-d/go-billy) [![Build status](https://ci.appveyor.com/api/projects/status/vx2qn6vlakbi724t?svg=true)](https://ci.appveyor.com/project/mcuadros/go-billy) [![codecov](https://codecov.io/gh/src-d/go-billy/branch/master/graph/badge.svg)](https://codecov.io/gh/src-d/go-billy)

The missing interface filesystem abstraction for Go.
Billy implements an interface based on the `os` standard library, allowing to develop applications without dependency on the underlying storage. Make virtually free implement an mocks and testing over filesystem operations.

Billy was born as part of [src-d/go-git](https://github.com/src-d/go-git) project.

## Installation

```go
go get -u gopkg.in/src-d/go-billy.v3/...
```

## Why billy?

The library billy deals with storage systems and Billy is the name of a well-known, IKEA
bookcase. That's it.

## Usage

Billy exposes filesystems using the
[`Filesystem` interface](https://godoc.org/github.com/src-d/go-billy#Filesystem).
Each filesystem implementation gives you a `New` method, whose arguments depend on
the implementation itself, that returns a new `Filesystem`.

The following example caches in memory all readable files in a directory from any
billy's filesystem implementation.

```go
func LoadToMemory(origin billy.Filesystem, path string) (*memory.Memory, error) {
	memory := memory.New()

	files, err := origin.ReadDir("/")
	if err != nil {
		return nil, err
	}

	for _, file := range files {
		if file.IsDir() {
			continue
		}

		src, err := origin.Open(file.Name())
		if err != nil {
			return nil, err
		}

		dst, err := memory.Create(file.Name())
		if err != nil {
			return nil, err
		}

		if _, err = io.Copy(dst, src); err != nil {
			return nil, err
		}

		if err := dst.Close(); err != nil {
			return nil, err
		}

		if err := src.Close(); err != nil {
			return nil, err
		}
	}

	return memory, nil
}
```

## License

MIT, see [LICENSE](LICENSE)
