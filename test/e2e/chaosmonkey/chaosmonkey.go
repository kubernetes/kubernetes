// chaosmonkeyjig is a test framework designed to allow e2e testing across a system while inducing a
// disruption.  It is designed to be modular enough that, when a dev is writing a new feature and
// wants to consider a disruptive event, they can run their own chaos monkey test with their feature.

// TODO(ihmccreery) Also, set precedent for factoring out into separate package, (it's going to be
// hard to avoid circular dependencies here).

package chaosmonkey

import (
	"fmt"
	"sync"

	. "github.com/onsi/ginkgo"
)

// An interface for chaos-monkey-testing a specific feature
type Interface interface {
	Setup() error
	// Test runs inside a goroutine, finishing when it receives a signal on the channel
	Test(chan int) error
	Teardown() error
}

type disruption func() error

type test struct {
	Interface
	setupErr    error
	testErr     error
	teardownErr error
	channel     chan int
}

// A jig for testing functionality across a disruptive event.
type Jig struct {
	disruption disruption
	tests      []test
}

func New(d disruption, ins ...Interface) *Jig {
	tests := make([]test, len(ins))
	for i, in := range ins {
		tests[i] = test{in, nil, nil, nil, nil}
	}
	return &Jig{d, tests}
}

func (j *Jig) Run() {
	var wg sync.WaitGroup
	failed := false

	// Run Setup for all registered tests, and wait to finish
	//
	// TODO(ihmccreery) Consider breaking this into a separate BeforeEach
	// call.
	for _, test := range j.tests {
		wg.Add(1)
		go func(t Interface) {
			defer wg.Done()
			test.setupErr = t.Setup()
		}(test)
	}
	wg.Wait()

	// go run Test for all registered tests
	for _, test := range j.tests {
		test.channel = make(chan int)
		wg.Add(1)
		go func(t Interface) {
			defer wg.Done()
			test.testErr = t.Test(test.channel)
		}(test)
	}
	// Trigger upgrade
	if err := j.disruption(); err != nil {
		// TODO(ihmccreery) Make this Logf, once it's factored into its
		// own package
		fmt.Fprintf(GinkgoWriter, err.Error())
	}
	// Once upgrade is done, signal to all Tests that upgrade is finished, and wait for all
	// tests to finish
	for _, test := range j.tests {
		test.channel <- 1
	}
	wg.Wait()

	// Call Teardown on all tests
	//
	// TODO(ihmccreery) Consider breaking this into a separate AfterEach
	// call.
	for _, test := range j.tests {
		wg.Add(1)
		go func(t Interface) {
			defer wg.Done()
			test.teardownErr = t.Teardown()
		}(test)
	}
	wg.Wait()

	// Collect & aggregate results of Tests, and report
	for _, test := range j.tests {
		if test.setupErr != nil{
			// TODO(ihmccreery) Make this Logf, once it's factored
			// into its own package
			fmt.Fprintf(GinkgoWriter, test.setupErr.Error())
			failed = true
		}
		if test.testErr != nil{
			// TODO(ihmccreery) Make this Logf, once it's factored
			// into its own package
			fmt.Fprintf(GinkgoWriter, test.testErr.Error())
			failed = true
		}
		if test.teardownErr != nil{
			// TODO(ihmccreery) Make this Logf, once it's factored
			// into its own package
			fmt.Fprintf(GinkgoWriter, test.teardownErr.Error())
			failed = true
		}
	}
	if failed {
		// TODO(ihmccreery) Make this Failf, once it's factored into
		// its own package
		Fail("At least one error encountered during test", 1)
	}
}
