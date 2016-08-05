# xurls

[![GoDoc](https://godoc.org/github.com/mvdan/xurls?status.svg)](https://godoc.org/github.com/mvdan/xurls) [![Travis](https://travis-ci.org/mvdan/xurls.svg?branch=master)](https://travis-ci.org/mvdan/xurls)

Extract urls from text using regular expressions.

	go get github.com/mvdan/xurls

```go
import "github.com/mvdan/xurls"

func main() {
	xurls.Relaxed.FindString("Do gophers live in golang.org?")
	// "golang.org"
	xurls.Relaxed.FindAllString("foo.com is http://foo.com/.", -1)
	// []string{"foo.com", "http://foo.com/"}
	xurls.Strict.FindAllString("foo.com is http://foo.com/.", -1)
	// []string{"http://foo.com/"}
}
```

#### cmd/xurls

Reads text and prints one url per line.

	go get github.com/mvdan/xurls/cmd/xurls

```shell
$ echo "Do gophers live in http://golang.org?" | xurls
http://golang.org
```
