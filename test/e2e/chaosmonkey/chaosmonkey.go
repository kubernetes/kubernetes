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
	. "github.com/onsi/gomega"
)

// An interface for chaos-monkey-testing a specific feature
type Interface interface {
	Setup() error
	// Test runs inside a goroutine, finishing when it receives a signal on the channel
	Test(<-chan struct{}) error
	Teardown() error
}

type disruption func() error

type test struct {
	name        string
	Interface
	setupErr    error
	testErr     error
	teardownErr error
	channel     chan struct{}
}

// A jig for testing functionality across a disruptive event.
type Jig struct {
	disruption disruption
	tests      []test
}

func New(d disruption) *Jig {
	return &Jig{d, []test{}}
}

func (j *Jig) Register(name string, in Interface) {
	j.tests = append(j.tests, test{name, in, nil, nil, nil, nil})
}

// Because upgrading is a destructive and long-running operation, we must setup
// and run tests all at once, before beginning the upgrade, then run the
// upgrade, then aggregate the tests.
//
// It's awkward to rely on Ginkgo's parallelism model because it doesn't have a
// good way of ensuring that all tests can be run simultaneously, which
// presents a risk of deadlock: if all tests aren't running simultaneously, we
// can't start the disruption.
// 
// However, because Ginkgo must compute test tree beforehand, we can't put any
// of the logic outside of BeforeEach or It.  So, we do the entire test inside
// of a sync.Once nested in a BeforeEach.
//
// This model does, however, place nice with Ginkgo's parallelism model,
// (though running these tests in parallel won't add much, just faster
// reporting): if more than one test runner is kicked off, the Its will block
// on BeforeEach, which itself blocks on the once.Do returning the first time.
func (j *Jig) Do() {
	var wg sync.WaitGroup
	var once sync.Once

	BeforeEach(func() {
		// No call to Do returns until the one call to f
		// returns, so this will block any Its from commencing
		once.Do(func() {
			// Run Setup for all registered tests, and wait
			// to finish
			for _, test := range j.tests {
				wg.Add(1)
				go func() {
					defer wg.Done()
					test.setupErr = test.Setup()
				}()
			}
			wg.Wait()

			// go run Test for all registered tests
			ch := make(chan struct{})
			for _, test := range j.tests {
				wg.Add(1)
				go func() {
					defer wg.Done()
					test.testErr = test.Test(ch)
				}()
			}
			// Trigger upgrade
			if err := j.disruption(); err != nil {
				// TODO(ihmccreery) Make this Logf,
				// once it's factored into its own
				// package
				fmt.Fprintf(GinkgoWriter, err.Error())
			}
			// Once upgrade is done, signal to all Tests
			// that upgrade is finished, and wait for all
			// tests to finish
			close(ch)
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
		})

		It("disruption succeeds", func() {
			// Check to make sure that the upgrade actually worked.
			// Not sure what the right abstraction here is.
		})
		for _, test := range j.tests {
			It(test.name, func() {
				// TODO(ihmccreery) Make this expectNoError,
				// once it's factored into its own package
				Expect(test.setupErr).NotTo(HaveOccurred())
				Expect(test.testErr).NotTo(HaveOccurred())
				Expect(test.teardownErr).NotTo(HaveOccurred())
			})
		}
	})
}
