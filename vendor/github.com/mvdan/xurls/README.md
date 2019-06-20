# xurls

[![GoDoc](https://godoc.org/github.com/mvdan/xurls?status.svg)](https://godoc.org/github.com/mvdan/xurls)
[![Travis](https://travis-ci.org/mvdan/xurls.svg?branch=master)](https://travis-ci.org/mvdan/xurls)

Extract urls from text using regular expressions.

	go get -u github.com/mvdan/xurls

```go
import "github.com/mvdan/xurls"

func main() {
	xurls.Relaxed.FindString("Do gophers live in golang.org?")
	// "golang.org"
	xurls.Strict.FindAllString("foo.com is http://foo.com/.", -1)
	// []string{"http://foo.com/"}
}
```

`Relaxed` is around five times slower than `Strict` since it does more
work to find the URLs without relying on the scheme:

```
BenchmarkStrictEmpty-4           1000000              1885 ns/op
BenchmarkStrictSingle-4           200000              8356 ns/op
BenchmarkStrictMany-4             100000             22547 ns/op
BenchmarkRelaxedEmpty-4           200000              7284 ns/op
BenchmarkRelaxedSingle-4           30000             58557 ns/op
BenchmarkRelaxedMany-4             10000            130251 ns/op
```

#### cmd/xurls

	go get -u github.com/mvdan/xurls/cmd/xurls

```shell
$ echo "Do gophers live in http://golang.org?" | xurls
http://golang.org
```
