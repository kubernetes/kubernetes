# go-fuzz-headers
This repository contains various helper functions for go fuzzing. It is mostly used in combination with [go-fuzz](https://github.com/dvyukov/go-fuzz), but compatibility with fuzzing in the standard library will also be supported. Any coverage guided fuzzing engine that provides an array or slice of bytes can be used with go-fuzz-headers.


## Usage
Using go-fuzz-headers is easy. First create a new consumer with the bytes provided by the fuzzing engine:

```go
import (
	fuzz "github.com/AdaLogics/go-fuzz-headers"
)
data := []byte{'R', 'a', 'n', 'd', 'o', 'm'}
f := fuzz.NewConsumer(data)

```

This creates a `Consumer` that consumes the bytes of the input as it uses them to fuzz different types.

After that, `f` can be used to easily create fuzzed instances of different types. Below are some examples:

### Structs
One of the most useful features of go-fuzz-headers is its ability to fill structs with the data provided by the fuzzing engine. This is done with a single line:
```go
type Person struct {
    Name string
    Age  int
}
p := Person{}
// Fill p with values based on the data provided by the fuzzing engine:
err := f.GenerateStruct(&p)
```

This includes nested structs too. In this example, the fuzz Consumer will also insert values in `p.BestFriend`:
```go
type PersonI struct {
    Name       string
    Age        int
    BestFriend PersonII
}
type PersonII struct {
    Name string
    Age  int
}
p := PersonI{}
err := f.GenerateStruct(&p)
```

If the consumer should insert values for unexported fields as well as exported, this can be enabled with:

```go
f.AllowUnexportedFields()
```

...and disabled with:

```go
f.DisallowUnexportedFields()
```

### Other types:

Other useful APIs:

```go
createdString, err := f.GetString() // Gets a string
createdInt, err := f.GetInt() // Gets an integer
createdByte, err := f.GetByte() // Gets a byte
createdBytes, err := f.GetBytes() // Gets a byte slice
createdBool, err := f.GetBool() // Gets a boolean
err := f.FuzzMap(target_map) // Fills a map
createdTarBytes, err := f.TarBytes() // Gets bytes of a valid tar archive
err := f.CreateFiles(inThisDir) // Fills inThisDir with files
createdString, err := f.GetStringFrom("anyCharInThisString", ofThisLength) // Gets a string that consists of chars from "anyCharInThisString" and has the exact length "ofThisLength"
```

Most APIs are added as they are needed.

## Projects that use go-fuzz-headers
- [runC](https://github.com/opencontainers/runc)
- [Istio](https://github.com/istio/istio)
- [Vitess](https://github.com/vitessio/vitess)
- [Containerd](https://github.com/containerd/containerd)

Feel free to add your own project to the list, if you use go-fuzz-headers to fuzz it.


 

## Status
The project is under development and will be updated regularly.

## References
go-fuzz-headers' approach to fuzzing structs is strongly inspired by [gofuzz](https://github.com/google/gofuzz).