go-colortext package [![GoSearch](http://go-search.org/badge?id=github.com%2Fdaviddengcn%2Fgo-colortext)](http://go-search.org/view?id=github.com%2Fdaviddengcn%2Fgo-colortext)
====================

This is a package to change the color of the text and background in the console, working both under Windows and other systems.

Under Windows, the console APIs are used. Otherwise, ANSI texts are output.

Docs: http://godoc.org/github.com/daviddengcn/go-colortext ([packages that import ct](http://go-search.org/view?id=github.com%2fdaviddengcn%2fgo-colortext))

Usage:
```go
ChangeColor(Red, true, White, false)
fmt.Println(...)
ChangeColor(Green, false, None, false)
fmt.Println(...)
ResetColor()
```
