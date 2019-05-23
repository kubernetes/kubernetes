//dedupimport -i

package pkg

// This is a copy of scope1.go
// with the -i flag.

import (
	"fmt"
	f "fmt"
	"math"
	m "math"
	"math/bits"
	x "math/bits"
)

type X interface {
	foo(fmt f.Formatter)
}

var s = func() {
	type fmt struct {
		D (f.Formatter)
	}
}

var f func() bool

type stringer struct {
	f.Stringer
}

func (fmt *stringer) qux() {
	{
		var bits = 0x0
		f.Sprint("frond")
		{
			{
				bits.OnesCount8()
			}
		}
	}
}

var count = x.Len

var myfunc = func(bits uint) bool {
	x.TrailingZeros(bits)
	return true
}

func namedReturnOK() (m int) {
	_ = m.Sin
	return
}

func namedReturnNotOK() (math int) {
	_ = m.Cos
	return
}
