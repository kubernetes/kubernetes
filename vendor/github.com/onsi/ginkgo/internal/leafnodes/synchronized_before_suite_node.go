package leafnodes

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"reflect"
	"time"

	"github.com/onsi/ginkgo/internal/failer"
	"github.com/onsi/ginkgo/types"
)

type synchronizedBeforeSuiteNode struct {
	runnerA *runner
	runnerB *runner

	data []byte

	outcome types.SpecState
	failure types.SpecFailure
	runTime time.Duration
}

func NewSynchronizedBeforeSuiteNode(bodyA interface{}, bodyB interface{}, codeLocation types.CodeLocation, timeout time.Duration, failer *failer.Failer) SuiteNode {
	node := &synchronizedBeforeSuiteNode{}

	node.runnerA = newRunner(node.wrapA(bodyA), codeLocation, timeout, failer, types.SpecComponentTypeBeforeSuite, 0)
	node.runnerB = newRunner(node.wrapB(bodyB), codeLocation, timeout, failer, types.SpecComponentTypeBeforeSuite, 0)

	return node
}

func (node *synchronizedBeforeSuiteNode) Run(parallelNode int, parallelTotal int, syncHost string) bool {
	t := time.Now()
	defer func() {
		node.runTime = time.Since(t)
	}()

	if parallelNode == 1 {
		node.outcome, node.failure = node.runA(parallelTotal, syncHost)
	} else {
		node.outcome, node.failure = node.waitForA(syncHost)
	}

	if node.outcome != types.SpecStatePassed {
		return false
	}
	node.outcome, node.failure = node.runnerB.run()

	return node.outcome == types.SpecStatePassed
}

func (node *synchronizedBeforeSuiteNode) runA(parallelTotal int, syncHost string) (types.SpecState, types.SpecFailure) {
	outcome, failure := node.runnerA.run()

	if parallelTotal > 1 {
		state := types.RemoteBeforeSuiteStatePassed
		if outcome != types.SpecStatePassed {
			state = types.RemoteBeforeSuiteStateFailed
		}
		json := (types.RemoteBeforeSuiteData{
			Data:  node.data,
			State: state,
		}).ToJSON()
		http.Post(syncHost+"/BeforeSuiteState", "application/json", bytes.NewBuffer(json))
	}

	return outcome, failure
}

func (node *synchronizedBeforeSuiteNode) waitForA(syncHost string) (types.SpecState, types.SpecFailure) {
	failure := func(message string) types.SpecFailure {
		return types.SpecFailure{
			Message:               message,
			Location:              node.runnerA.codeLocation,
			ComponentType:         node.runnerA.nodeType,
			ComponentIndex:        node.runnerA.componentIndex,
			ComponentCodeLocation: node.runnerA.codeLocation,
		}
	}
	for {
		resp, err := http.Get(syncHost + "/BeforeSuiteState")
		if err != nil || resp.StatusCode != http.StatusOK {
			return types.SpecStateFailed, failure("Failed to fetch BeforeSuite state")
		}

		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return types.SpecStateFailed, failure("Failed to read BeforeSuite state")
		}
		resp.Body.Close()

		beforeSuiteData := types.RemoteBeforeSuiteData{}
		err = json.Unmarshal(body, &beforeSuiteData)
		if err != nil {
			return types.SpecStateFailed, failure("Failed to decode BeforeSuite state")
		}

		switch beforeSuiteData.State {
		case types.RemoteBeforeSuiteStatePassed:
			node.data = beforeSuiteData.Data
			return types.SpecStatePassed, types.SpecFailure{}
		case types.RemoteBeforeSuiteStateFailed:
			return types.SpecStateFailed, failure("BeforeSuite on Node 1 failed")
		case types.RemoteBeforeSuiteStateDisappeared:
			return types.SpecStateFailed, failure("Node 1 disappeared before completing BeforeSuite")
		}

		time.Sleep(50 * time.Millisecond)
	}
}

func (node *synchronizedBeforeSuiteNode) Passed() bool {
	return node.outcome == types.SpecStatePassed
}

func (node *synchronizedBeforeSuiteNode) Summary() *types.SetupSummary {
	return &types.SetupSummary{
		ComponentType: node.runnerA.nodeType,
		CodeLocation:  node.runnerA.codeLocation,
		State:         node.outcome,
		RunTime:       node.runTime,
		Failure:       node.failure,
	}
}

func (node *synchronizedBeforeSuiteNode) wrapA(bodyA interface{}) interface{} {
	typeA := reflect.TypeOf(bodyA)
	if typeA.Kind() != reflect.Func {
		panic("SynchronizedBeforeSuite expects a function as its first argument")
	}

	takesNothing := typeA.NumIn() == 0
	takesADoneChannel := typeA.NumIn() == 1 && typeA.In(0).Kind() == reflect.Chan && typeA.In(0).Elem().Kind() == reflect.Interface
	returnsBytes := typeA.NumOut() == 1 && typeA.Out(0).Kind() == reflect.Slice && typeA.Out(0).Elem().Kind() == reflect.Uint8

	if !((takesNothing || takesADoneChannel) && returnsBytes) {
		panic("SynchronizedBeforeSuite's first argument should be a function that returns []byte and either takes no arguments or takes a Done channel.")
	}

	if takesADoneChannel {
		return func(done chan<- interface{}) {
			out := reflect.ValueOf(bodyA).Call([]reflect.Value{reflect.ValueOf(done)})
			node.data = out[0].Interface().([]byte)
		}
	}

	return func() {
		out := reflect.ValueOf(bodyA).Call([]reflect.Value{})
		node.data = out[0].Interface().([]byte)
	}
}

func (node *synchronizedBeforeSuiteNode) wrapB(bodyB interface{}) interface{} {
	typeB := reflect.TypeOf(bodyB)
	if typeB.Kind() != reflect.Func {
		panic("SynchronizedBeforeSuite expects a function as its second argument")
	}

	returnsNothing := typeB.NumOut() == 0
	takesBytesOnly := typeB.NumIn() == 1 && typeB.In(0).Kind() == reflect.Slice && typeB.In(0).Elem().Kind() == reflect.Uint8
	takesBytesAndDone := typeB.NumIn() == 2 &&
		typeB.In(0).Kind() == reflect.Slice && typeB.In(0).Elem().Kind() == reflect.Uint8 &&
		typeB.In(1).Kind() == reflect.Chan && typeB.In(1).Elem().Kind() == reflect.Interface

	if !((takesBytesOnly || takesBytesAndDone) && returnsNothing) {
		panic("SynchronizedBeforeSuite's second argument should be a function that returns nothing and either takes []byte or ([]byte, Done)")
	}

	if takesBytesAndDone {
		return func(done chan<- interface{}) {
			reflect.ValueOf(bodyB).Call([]reflect.Value{reflect.ValueOf(node.data), reflect.ValueOf(done)})
		}
	}

	return func() {
		reflect.ValueOf(bodyB).Call([]reflect.Value{reflect.ValueOf(node.data)})
	}
}
