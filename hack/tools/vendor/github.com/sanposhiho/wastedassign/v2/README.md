# wastedassign
`wastedassign` finds wasted assignment statements

found the value ...

- reassigned, but never used afterward
- reassigned, but reassigned without using the value

## Example

```go
package main

import "fmt"

func f() int {
	a := 0 
        b := 0
        fmt.Print(a)
        fmt.Print(b)
        a = 1  // This reassignment is wasted, because never used afterwards. Wastedassign find this 

        b = 1  // This reassignment is wasted, because reassigned without use this value. Wastedassign find this 
        b = 2
        fmt.Print(b)
        
	return 1 + 2
}
```


```bash
$ go vet -vettool=`which wastedassign` sample.go            
# command-line-arguments
./sample.go:10:2: assigned to a, but never used afterwards
./sample.go:12:2: assigned to b, but reassigned without using the value
```


## Installation

```
go get -u github.com/sanposhiho/wastedassign/v2/cmd/wastedassign
```

## Usage

```
# in your project

go vet -vettool=`which wastedassign` ./...
```

And, you can use wastedassign in [golangci-lint](https://github.com/golangci/golangci-lint).

## Contribution

I am waiting for your contribution :D 

Feel free to create an issue or a PR!

### Run test

```
go test
```
