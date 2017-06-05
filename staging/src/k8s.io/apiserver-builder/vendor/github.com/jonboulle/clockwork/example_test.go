package clockwork

import (
	"sync"
	"testing"
	"time"
)

// my_func is an example of a time-dependent function, using an
// injected clock
func my_func(clock Clock, i *int) {
	clock.Sleep(3 * time.Second)
	*i += 1
}

// assert_state is an example of a state assertion in a test
func assert_state(t *testing.T, i, j int) {
	if i != j {
		t.Fatalf("i %d, j %d", i, j)
	}
}

// TestMyFunc tests my_func's behaviour with a FakeClock
func TestMyFunc(t *testing.T) {
	var i int
	c := NewFakeClock()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		my_func(c, &i)
		wg.Done()
	}()

	// Wait until my_func is actually sleeping on the clock
	c.BlockUntil(1)

	// Assert the initial state
	assert_state(t, i, 0)

	// Now advance the clock forward in time
	c.Advance(1 * time.Hour)

	// Wait until the function completes
	wg.Wait()

	// Assert the final state
	assert_state(t, i, 1)
}
