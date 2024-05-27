//go:build (!linux && !darwin && !dragonfly && !freebsd && !netbsd && !solaris) || (!amd64 && !arm64 && !mips64x && !ppc64 && !ppc64le && !loong64 && !s390x)

package starlark

// generic Int implementation as a union

import "math/big"

type intImpl struct {
	// We use only the signed 32-bit range of small to ensure
	// that small+small and small*small do not overflow.
	small_ int64    // minint32 <= small <= maxint32
	big_   *big.Int // big != nil <=> value is not representable as int32
}

// --- low-level accessors ---

// get returns the small and big components of the Int.
// small is defined only if big is nil.
// small is sign-extended to 64 bits for ease of subsequent arithmetic.
func (i Int) get() (small int64, big *big.Int) {
	return i.impl.small_, i.impl.big_
}

// Precondition: math.MinInt32 <= x && x <= math.MaxInt32
func makeSmallInt(x int64) Int {
	return Int{intImpl{small_: x}}
}

// Precondition: x cannot be represented as int32.
func makeBigInt(x *big.Int) Int {
	return Int{intImpl{big_: x}}
}
