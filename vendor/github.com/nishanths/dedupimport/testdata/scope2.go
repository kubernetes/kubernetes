package pkg

import (
	"fmt"
	f "fmt"
)

func s() {
	type fmt struct{}
}

type stringer struct {
	f.Stringer
}
