package internal

import (
	"fmt"
	"sync"

	"github.com/onsi/ginkgo/v2/types"
)

type Failer struct {
	lock    *sync.Mutex
	failure types.Failure
	state   types.SpecState
}

func NewFailer() *Failer {
	return &Failer{
		lock:  &sync.Mutex{},
		state: types.SpecStatePassed,
	}
}

func (f *Failer) GetState() types.SpecState {
	f.lock.Lock()
	defer f.lock.Unlock()
	return f.state
}

func (f *Failer) GetFailure() types.Failure {
	f.lock.Lock()
	defer f.lock.Unlock()
	return f.failure
}

func (f *Failer) Panic(location types.CodeLocation, forwardedPanic interface{}) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.state == types.SpecStatePassed {
		f.state = types.SpecStatePanicked
		f.failure = types.Failure{
			Message:        "Test Panicked",
			Location:       location,
			ForwardedPanic: fmt.Sprintf("%v", forwardedPanic),
		}
	}
}

func (f *Failer) Fail(message string, location types.CodeLocation) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.state == types.SpecStatePassed {
		f.state = types.SpecStateFailed
		f.failure = types.Failure{
			Message:  message,
			Location: location,
		}
	}
}

func (f *Failer) Skip(message string, location types.CodeLocation) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.state == types.SpecStatePassed {
		f.state = types.SpecStateSkipped
		f.failure = types.Failure{
			Message:  message,
			Location: location,
		}
	}
}

func (f *Failer) AbortSuite(message string, location types.CodeLocation) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.state == types.SpecStatePassed {
		f.state = types.SpecStateAborted
		f.failure = types.Failure{
			Message:  message,
			Location: location,
		}
	}
}

func (f *Failer) Drain() (types.SpecState, types.Failure) {
	f.lock.Lock()
	defer f.lock.Unlock()

	failure := f.failure
	outcome := f.state

	f.state = types.SpecStatePassed
	f.failure = types.Failure{}

	return outcome, failure
}
