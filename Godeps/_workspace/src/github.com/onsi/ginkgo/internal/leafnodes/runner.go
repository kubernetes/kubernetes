package leafnodes

import (
	"fmt"
	"github.com/onsi/ginkgo/internal/codelocation"
	"github.com/onsi/ginkgo/internal/failer"
	"github.com/onsi/ginkgo/types"
	"reflect"
	"time"
)

type runner struct {
	isAsync          bool
	asyncFunc        func(chan<- interface{})
	syncFunc         func()
	codeLocation     types.CodeLocation
	timeoutThreshold time.Duration
	nodeType         types.SpecComponentType
	componentIndex   int
	failer           *failer.Failer
}

func newRunner(body interface{}, codeLocation types.CodeLocation, timeout time.Duration, failer *failer.Failer, nodeType types.SpecComponentType, componentIndex int) *runner {
	bodyType := reflect.TypeOf(body)
	if bodyType.Kind() != reflect.Func {
		panic(fmt.Sprintf("Expected a function but got something else at %v", codeLocation))
	}

	runner := &runner{
		codeLocation:     codeLocation,
		timeoutThreshold: timeout,
		failer:           failer,
		nodeType:         nodeType,
		componentIndex:   componentIndex,
	}

	switch bodyType.NumIn() {
	case 0:
		runner.syncFunc = body.(func())
		return runner
	case 1:
		if !(bodyType.In(0).Kind() == reflect.Chan && bodyType.In(0).Elem().Kind() == reflect.Interface) {
			panic(fmt.Sprintf("Must pass a Done channel to function at %v", codeLocation))
		}

		wrappedBody := func(done chan<- interface{}) {
			bodyValue := reflect.ValueOf(body)
			bodyValue.Call([]reflect.Value{reflect.ValueOf(done)})
		}

		runner.isAsync = true
		runner.asyncFunc = wrappedBody
		return runner
	}

	panic(fmt.Sprintf("Too many arguments to function at %v", codeLocation))
}

func (r *runner) run() (outcome types.SpecState, failure types.SpecFailure) {
	if r.isAsync {
		return r.runAsync()
	} else {
		return r.runSync()
	}
}

func (r *runner) runAsync() (outcome types.SpecState, failure types.SpecFailure) {
	done := make(chan interface{}, 1)

	go func() {
		finished := false

		defer func() {
			if e := recover(); e != nil || !finished {
				r.failer.Panic(codelocation.New(2), e)
				select {
				case <-done:
					break
				default:
					close(done)
				}
			}
		}()

		r.asyncFunc(done)
		finished = true
	}()

	select {
	case <-done:
	case <-time.After(r.timeoutThreshold):
		r.failer.Timeout(r.codeLocation)
	}

	failure, outcome = r.failer.Drain(r.nodeType, r.componentIndex, r.codeLocation)
	return
}
func (r *runner) runSync() (outcome types.SpecState, failure types.SpecFailure) {
	finished := false

	defer func() {
		if e := recover(); e != nil || !finished {
			r.failer.Panic(codelocation.New(2), e)
		}

		failure, outcome = r.failer.Drain(r.nodeType, r.componentIndex, r.codeLocation)
	}()

	r.syncFunc()
	finished = true

	return
}
