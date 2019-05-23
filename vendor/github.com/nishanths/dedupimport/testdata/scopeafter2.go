package main

import (
	"fmt"
	f "fmt"
)

var _ fmt.Stringer // just to have an existing use of fmt import

func main() {
	{
		// should rewrite f -> fmt; the fmt var declaration comes after,
		// so it is safe to rewrite.
		f.Println("Hello, playground")
	}
	var fmt string
	_ = fmt
}
