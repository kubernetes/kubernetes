package base

import (
	"sync"
	"testing"
	"time"
)

func TestRegulatorEnterExit(t *testing.T) {
	const limit = 500

	r := NewRegulator(nil, limit).(*regulator)

	for try := 0; try < 50; try++ {
		run := make(chan struct{})

		var firstGroupReady sync.WaitGroup
		var firstGroupDone sync.WaitGroup
		firstGroupReady.Add(limit)
		firstGroupDone.Add(limit)
		for i := 0; i < limit; i++ {
			go func() {
				r.enter()
				firstGroupReady.Done()
				<-run
				r.exit()
				firstGroupDone.Done()
			}()
		}
		firstGroupReady.Wait()

		// now we exhausted all the limit, let's run a little bit more
		var secondGroupReady sync.WaitGroup
		var secondGroupDone sync.WaitGroup
		for i := 0; i < 50; i++ {
			secondGroupReady.Add(1)
			secondGroupDone.Add(1)
			go func() {
				secondGroupReady.Done()
				r.enter()
				r.exit()
				secondGroupDone.Done()
			}()
		}
		secondGroupReady.Wait()

		// allow the first group to return resources
		close(run)

		done := make(chan struct{})
		go func() {
			secondGroupDone.Wait()
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("some r.enter() are still locked")
		}

		firstGroupDone.Wait()

		if r.available != limit {
			t.Fatalf("r.available: got %d, want %d", r.available, limit)
		}
	}
}
