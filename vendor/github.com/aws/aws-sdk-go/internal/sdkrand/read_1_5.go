// +build !go1.6

package sdkrand

import "math/rand"

// Read backfills Go 1.6's math.Rand.Reader for Go 1.5
func Read(r *rand.Rand, p []byte) (n int, err error) {
	// Copy of Go standard libraries math package's read function not added to
	// standard library until Go 1.6.
	var pos int8
	var val int64
	for n = 0; n < len(p); n++ {
		if pos == 0 {
			val = r.Int63()
			pos = 7
		}
		p[n] = byte(val)
		val >>= 8
		pos--
	}

	return n, err
}
