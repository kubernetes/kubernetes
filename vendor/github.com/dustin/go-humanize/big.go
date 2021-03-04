package humanize

import (
	"math/big"
)

// order of magnitude (to a max order)
func oomm(n, b *big.Int, maxmag int) (float64, int) {
	mag := 0
	m := &big.Int{}
	for n.Cmp(b) >= 0 {
		n.DivMod(n, b, m)
		mag++
		if mag == maxmag && maxmag >= 0 {
			break
		}
	}
	return float64(n.Int64()) + (float64(m.Int64()) / float64(b.Int64())), mag
}

// total order of magnitude
// (same as above, but with no upper limit)
func oom(n, b *big.Int) (float64, int) {
	mag := 0
	m := &big.Int{}
	for n.Cmp(b) >= 0 {
		n.DivMod(n, b, m)
		mag++
	}
	return float64(n.Int64()) + (float64(m.Int64()) / float64(b.Int64())), mag
}
