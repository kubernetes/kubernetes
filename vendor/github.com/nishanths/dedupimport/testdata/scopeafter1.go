package pkg

// Original built and running: https://play.golang.org/p/32aD9NLoN4
// Rewritten output built and running: https://play.golang.org/p/gcQGANBuEM

import (
	"fmt"
	f "fmt"
)

var _ fmt.Stringer // just to have an existing use of fmt import

func foo() {
	// should rewrite f -> fmt; the fmt var declaration comes after,
	// so it is safe to rewrite.
	f.Println("Hello, playground")
	var fmt string
	fmt = "x"
	println("woo" + fmt)
}
