/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package chaosmonkey

import . "github.com/onsi/ginkgo"

// Disruption is the type to construct a chaosmonkey with; see Do for more information.
type Disruption func()

// Test is the type to register with a chaosmonkey.  A test will run asynchronously across the
// chaosmonkey's Disruption.  A Test takes a Semaphore as an argument.  It should call sem.Ready()
// once it's ready for the disruption to start and should then wait until sem.StopCh (which is a
// <-chan struct{}) is closed, which signals that the disruption is over.  It should then clean up
// and return.  See Do and Semaphore for more information.
type Test func(sem *Semaphore)

// Interface can be implemented if you prefer to define tests without dealing with a Semaphore.  You
// may define a struct that implements Interface's three methods (Setup, Test, and Teardown) and
// RegisterInterface.  See RegisterInterface for more information.
type Interface interface {
	Setup()
	Test(stopCh <-chan struct{})
	Teardown()
}

type chaosmonkey struct {
	disruption Disruption
	tests      []Test
}

// New creates and returns a chaosmonkey, with which the caller should register Tests and call Do.
// See Do for more information.
func New(disruption Disruption) *chaosmonkey {
	return &chaosmonkey{
		disruption,
		[]Test{},
	}
}

// Register registers the given Test with the chaosmonkey, so that the test will run over the
// Disruption.
func (cm *chaosmonkey) Register(test Test) {
	cm.tests = append(cm.tests, test)
}

// RegisterInterface registers the given Interface with the chaosmonkey, so the chaosmonkey will
// call Setup, Test, and Teardown properly.  Test can tell that the Disruption is finished when
// stopCh is closed.
func (cm *chaosmonkey) RegisterInterface(in Interface) {
	cm.Register(func(sem *Semaphore) {
		in.Setup()
		sem.Ready()
		in.Test(sem.StopCh)
		in.Teardown()
	})
}

// Do performs the Disruption while testing the registered Tests.  Once the caller has registered
// all Tests with the chaosmonkey, they call Do.  Do starts each registered test asynchronously and
// waits for each test to signal that it is ready by calling sem.Ready().  Do will then do the
// Disruption, and when it's complete, close sem.StopCh to signal to the registered Tests that the
// Disruption is over, and wait for all Tests to return.
func (cm *chaosmonkey) Do() {
	sems := []*Semaphore{}
	// All semaphores have the same StopCh.
	stopCh := make(chan struct{})

	for _, test := range cm.tests {
		test := test
		sem := newSemaphore(stopCh)
		sems = append(sems, sem)
		go func() {
			defer GinkgoRecover()
			defer sem.done()
			test(sem)
		}()
	}

	By("Waiting for all async tests to be ready")
	for _, sem := range sems {
		// Wait for test to be ready.  We have to wait for ready *or done* because a test
		// may panic before signaling that its ready, and we shouldn't block.  Since we
		// defered sem.done() above, if a test panics, it's marked as done.
		sem.waitForReadyOrDone()
	}

	By("Starting disruption")
	cm.disruption()
	By("Disruption complete; stopping async validations")
	close(stopCh)
	By("Waiting for async validations to complete")
	for _, sem := range sems {
		sem.waitForDone()
	}
}

// Semaphore is taken by a Test and provides: Ready(), for the Test to call when it's ready for the
// disruption to start; and StopCh, the closure of which signals to the Test that the disruption is
// finished.
type Semaphore struct {
	readyCh chan struct{}
	StopCh  <-chan struct{}
	doneCh  chan struct{}
}

func newSemaphore(stopCh <-chan struct{}) *Semaphore {
	// We don't want to block on Ready() or done()
	return &Semaphore{
		make(chan struct{}, 1),
		stopCh,
		make(chan struct{}, 1),
	}
}

// Ready is called by the Test to signal that the Test is ready for the disruption to start.
func (sem *Semaphore) Ready() {
	close(sem.readyCh)
}

// done is an internal method for Go to defer, both to wait for all tests to return, but also to
// sense if a test panicked before calling Ready.  See waitForReadyOrDone.
func (sem *Semaphore) done() {
	close(sem.doneCh)
}

// We would like to just check if all tests are ready, but if they fail (which Ginkgo implements as
// a panic), they may not have called Ready().  We check done as well to see if the function has
// already returned; if it has, we don't care if it's ready, and just continue.
func (sem *Semaphore) waitForReadyOrDone() {
	select {
	case <-sem.readyCh:
	case <-sem.doneCh:
	}
}

// waitForDone is an internal method for Go to wait on all Tests returning.
func (sem *Semaphore) waitForDone() {
	<-sem.doneCh
}
