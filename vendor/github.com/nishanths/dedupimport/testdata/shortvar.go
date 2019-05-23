package pkg

import f "fmt"
import "fmt"

var _ fmt.Stringer

func foo() {
	fmt := "x"
	f.Println("go!")
}