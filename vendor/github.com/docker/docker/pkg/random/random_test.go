package random

import (
	"math/rand"
	"sync"
	"testing"
)

// for go test -v -race
func TestConcurrency(t *testing.T) {
	rnd := rand.New(NewSource())
	var wg sync.WaitGroup

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			rnd.Int63()
			wg.Done()
		}()
	}
	wg.Wait()
}
