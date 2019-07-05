package failer

import (
	"fmt"
	"sync"

	"github.com/onsi/ginkgo/types"
)

type Failer struct {
	lock    *sync.Mutex
	failure types.SpecFailure
	state   types.SpecState
}

func New() *Failer {
	return &Failer{
		lock:  &sync.Mutex{},
		state: types.SpecStatePassed,
	}
}

func (f *Failer) Panic(location types.CodeLocation, forwardedPanic interface{}) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.state == types.SpecStatePassed {
		f.state = types.SpecStatePanicked
		f.failure = types.SpecFailure{
			Message:        "Test Panicked",
			Location:       location,
			ForwardedPanic: fmt.Sprintf("%v", forwardedPanic),
		}
	}
}

func (f *Failer) Timeout(location types.CodeLocation) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.state == types.SpecStatePassed {
		f.state = types.SpecStateTimedOut
		f.failure = types.SpecFailure{
			Message:  "Timed out",
			Location: location,
		}
	}
}

func (f *Failer) Fail(message string, location types.CodeLocation) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.state == types.SpecStatePassed {
		f.state = types.SpecStateFailed
		f.failure = types.SpecFailure{
			Message:  message,
			Location: location,
		}
	}
}

func (f *Failer) Drain(componentType types.SpecComponentType, componentIndex int, componentCodeLocation types.CodeLocation) (types.SpecFailure, types.SpecState) {
	f.lock.Lock()
	defer f.lock.Unlock()

	failure := f.failure
	outcome := f.state
	if outcome != types.SpecStatePassed {
		failure.ComponentType = componentType
		failure.ComponentIndex = componentIndex
		failure.ComponentCodeLocation = componentCodeLocation
	}

	f.state = types.SpecStatePassed
	f.failure = types.SpecFailure{}

	return failure, outcome
}

func (f *Failer) Skip(message string, location types.CodeLocation) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.state == types.SpecStatePassed {
		f.state = types.SpecStateSkipped
		f.failure = types.SpecFailure{
			Message:  message,
			Location: location,
		}
	}
}
