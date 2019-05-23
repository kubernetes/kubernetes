package pkg

// this would be a compile error:
//  bits redeclared as imported package name
// but we have to resolve the duplicates anyway.
import (
	bits "math/bits"
	"math/bits"
)

func bar() {
	_ = bits.Len16(0)
}