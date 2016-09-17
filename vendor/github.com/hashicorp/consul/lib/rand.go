package lib

import (
	crand "crypto/rand"
	"math"
	"math/big"
	"math/rand"
	"sync"
	"time"
)

var (
	once sync.Once

	// SeededSecurely is set to true if a cryptographically secure seed
	// was used to initialize rand.  When false, the start time is used
	// as a seed.
	SeededSecurely bool
)

// SeedMathRand provides weak, but guaranteed seeding, which is better than
// running with Go's default seed of 1.  A call to SeedMathRand() is expected
// to be called via init(), but never a second time.
func SeedMathRand() {
	once.Do(func() {
		n, err := crand.Int(crand.Reader, big.NewInt(math.MaxInt64))
		if err != nil {
			rand.Seed(time.Now().UTC().UnixNano())
			return
		}
		rand.Seed(n.Int64())
		SeededSecurely = true
	})
}
