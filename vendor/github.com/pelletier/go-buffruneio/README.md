# buffruneio

[![Tests Status](https://travis-ci.org/pelletier/go-buffruneio.svg?branch=master)](https://travis-ci.org/pelletier/go-buffruneio)
[![GoDoc](https://godoc.org/github.com/pelletier/go-buffruneio?status.svg)](https://godoc.org/github.com/pelletier/go-buffruneio)

Buffruneio is a wrapper around bufio to provide buffered runes access with
unlimited unreads.

```go
import "github.com/pelletier/go-buffruneio"
```

## Examples

```go
import (
    "fmt"
    "github.com/pelletier/go-buffruneio"
    "strings"
)

reader := buffruneio.NewReader(strings.NewReader("abcd"))
fmt.Println(reader.ReadRune()) // 'a'
fmt.Println(reader.ReadRune()) // 'b'
fmt.Println(reader.ReadRune()) // 'c'
reader.UnreadRune()
reader.UnreadRune()
fmt.Println(reader.ReadRune()) // 'b'
fmt.Println(reader.ReadRune()) // 'c'
```

## Documentation

The documentation and additional examples are available at
[godoc.org](http://godoc.org/github.com/pelletier/go-buffruneio).

## Contribute

Feel free to report bugs and patches using GitHub's pull requests system on
[pelletier/go-toml](https://github.com/pelletier/go-buffruneio). Any feedback is
much appreciated!

## LICENSE

Copyright (c) 2016 Thomas Pelletier

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
