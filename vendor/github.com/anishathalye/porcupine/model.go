package porcupine

import (
	"context"
	"fmt"
	"strings"
)

// An Operation is an element of a history.
//
// This package supports two different representations of histories, as a
// sequence of Operation or [Event]. In the Operation representation, function
// call/returns are packaged together, along with timestamps of when the
// function call was made and when the function call returned.
//
// The interval [Call, Return] is interpreted as a closed interval, so an
// operation with interval [10, 20] is concurrent with another operation with
// interval [20, 30].
type Operation struct {
	ClientId int // optional, unless you want a visualization; zero-indexed
	Input    interface{}
	Call     int64 // invocation timestamp
	Output   interface{}
	Return   int64 // response timestamp
	// Metadata contains arbitrary metadata associated with the operation.
	// It is not used for linearizability checking but can be used for visualization.
	Metadata interface{}
	_        struct{} // disallow positional literals, for extensibility
}

// Interpreting the interval [Call, Return] as a closed interval is the only
// reasonable approach for how we expect this library to be used. Otherwise, we
// might have the following situation, when using a monotonic clock to get
// timestamps (where successive calls to the clock return values that are always
// greater than _or equal to_ previously returned values):
//
// - Client 1 calls clock(), gets ts=1, invokes put("x", "y")
// - Client 2 calls clock(), gets ts=2, invokes get("x")
// - Client 1 operation returns, calls clock(), gets ts=2
// - Client 2 operation returns "", calls clock(), gets ts=3
//
// These operations were concurrent, but if we interpret the intervals as
// half-open, for example, Client 1's operation had interval [1, 2) and Client
// 2's operation had interval [2, 3), so they are not concurrent operations, and
// we'd say that this history is not linearizable, which is not correct. The
// only sensible approach is to interpret the interval [Call, Return] as a
// closed interval.

// An EventKind tags an [Event] as either a function call or a return.
type EventKind bool

const (
	CallEvent   EventKind = false
	ReturnEvent EventKind = true
)

// An Event is an element of a history, a function call event or a return
// event.
//
// This package supports two different representations of histories, as a
// sequence of Event or [Operation]. In the Event representation, function
// calls and returns are only relatively ordered and do not have absolute
// timestamps.
//
// The Id field is used to match a function call event with its corresponding
// return event.
type Event struct {
	ClientId int // optional, unless you want a visualization; zero-indexed
	Kind     EventKind
	Value    interface{}
	Id       int
	// Metadata contains arbitrary metadata associated with the operation.
	// It is not used for linearizability checking but can be used for visualization.
	// Can be set on CallEvent or ReturnEvent. If both have metadata, ReturnEvent metadata takes precedence.
	Metadata interface{}
	_        struct{} // disallow positional literals, for extensibility
}

// A Model is a sequential specification of a system.
//
// Note: models in this package are expected to be purely functional. That is,
// the model Step function should not modify the given state (or input or
// output), but return a new state.
//
// Only the Init, Step (or StepContext), and Equal functions are necessary to
// specify if you just want to test histories for linearizability.
//
// Implementing the partition functions can greatly improve performance. If
// you're implementing the partition function, the model Init and Step
// functions can be per-partition. For example, if your specification is for a
// key-value store and you partition by key, then the per-partition state
// representation can just be a single value rather than a map.
//
// Implementing DescribeOperation and DescribeState will produce nicer
// visualizations.
//
// It may be helpful to look at this package's [test code] for examples of how
// to write models, including models that include partition functions.
//
// [test code]: https://github.com/anishathalye/porcupine/blob/master/porcupine_test.go
type Model struct {
	// Partition functions, such that a history is linearizable if and only
	// if each partition is linearizable. If left nil, this package will
	// skip partitioning.
	Partition      func(history []Operation) [][]Operation
	PartitionEvent func(history []Event) [][]Event
	// Initial state of the system.
	Init func() interface{}
	// Step function for the system. Returns whether or not the system
	// could take this step with the given inputs and outputs and also
	// returns the new state. This function must be a pure function: it
	// cannot mutate the given state.
	Step func(state interface{}, input interface{}, output interface{}) (bool, interface{})
	// StepContext is an optional context-aware Step function. If set, the
	// checker and visualization replay call StepContext instead of Step,
	// allowing the model to stop work promptly when a timeout expires.
	StepContext func(ctx context.Context, state interface{}, input interface{}, output interface{}) (bool, interface{})
	// Equality on states. If left nil, this package will use == as a
	// fallback ([ShallowEqual]).
	Equal func(state1, state2 interface{}) bool
	// Hash returns a hash of the given state. If non-nil, the checker uses
	// it to reduce the number of [Equal] comparisons. Must satisfy the
	// invariant that [Equal] states have equal hashes.
	Hash func(state interface{}) uint64
	// For visualization, describe an operation as a string. For example,
	// "Get('x') -> 'y'". Can be omitted if you're not producing
	// visualizations.
	DescribeOperation func(input interface{}, output interface{}) string
	// For visualization purposes, describe a state as a string. For
	// example, "{'x' -> 'y', 'z' -> 'w'}". Can be omitted if you're not
	// producing visualizations.
	DescribeState func(state interface{}) string
	// For visualization purposes, describe metadata as a string. Can be
	// omitted if you're not producing visualizations.
	DescribeOperationMetadata func(info interface{}) string
	_                         struct{} // disallow positional literals, for extensibility
}

// A NondeterministicModel is a nondeterministic sequential specification of a
// system.
//
// For basics on models, see the documentation for [Model].  In contrast to
// Model, NondeterministicModel has a step function that returns a set of
// states, indicating all possible next states. It can be converted to a Model
// using the [NondeterministicModel.ToModel] function.
//
// It may be helpful to look at this package's [test code] for examples of how
// to write and use nondeterministic models.
//
// [test code]: https://github.com/anishathalye/porcupine/blob/master/porcupine_test.go
type NondeterministicModel struct {
	// Partition functions, such that a history is linearizable if and only
	// if each partition is linearizable. If left nil, this package will
	// skip partitioning.
	Partition      func(history []Operation) [][]Operation
	PartitionEvent func(history []Event) [][]Event
	// Initial states of the system.
	Init func() []interface{}
	// Step function for the system. Returns all possible next states for
	// the given state, input, and output. If the system cannot step with
	// the given state/input to produce the given output, this function
	// should return an empty slice.
	Step func(state interface{}, input interface{}, output interface{}) []interface{}
	// StepContext is an optional context-aware Step function. If set, the
	// checker calls StepContext instead of Step, allowing the model to stop
	// work promptly when a timeout expires.
	StepContext func(ctx context.Context, state interface{}, input interface{}, output interface{}) []interface{}
	// Equality on states. If left nil, this package will use == as a
	// fallback ([ShallowEqual]).
	Equal func(state1, state2 interface{}) bool
	// Hash returns a hash of the given state. If non-nil, the checker uses
	// it to reduce the number of [Equal] comparisons. Must satisfy the
	// invariant that [Equal] states have equal hashes.
	Hash func(state interface{}) uint64
	// For visualization, describe an operation as a string. For example,
	// "Get('x') -> 'y'". Can be omitted if you're not producing
	// visualizations.
	DescribeOperation func(input interface{}, output interface{}) string
	// For visualization purposes, describe a state as a string. For
	// example, "{'x' -> 'y', 'z' -> 'w'}". Can be omitted if you're not
	// producing visualizations.
	DescribeState func(state interface{}) string
	// For visualization purposes, describe metadata as a string. Can be
	// omitted if you're not producing visualizations.
	DescribeOperationMetadata func(info interface{}) string
	_                         struct{} // disallow positional literals, for extensibility
}

func merge(states []interface{}, eq func(state1, state2 interface{}) bool) []interface{} {
	var uniqueStates []interface{}
	for _, state := range states {
		unique := true
		for _, us := range uniqueStates {
			if eq(state, us) {
				unique = false
				break
			}
		}
		if unique {
			uniqueStates = append(uniqueStates, state)
		}
	}
	return uniqueStates
}

// ToModel converts a [NondeterministicModel] to a [Model] using a power set
// construction.
//
// This makes it suitable for use in linearizability checking operations like
// [CheckOperations]. This is a general construction that can be used for any
// nondeterministic model. It relies on the NondeterministicModel's Equal
// function to merge states. You may be able to achieve better performance by
// implementing a Model directly.
func (nm *NondeterministicModel) ToModel() Model {
	// like fillDefault
	equal := nm.Equal
	if equal == nil {
		equal = shallowEqual
	}
	describeOperation := nm.DescribeOperation
	if describeOperation == nil {
		describeOperation = defaultDescribeOperation
	}
	describeState := nm.DescribeState
	if describeState == nil {
		describeState = defaultDescribeState
	}
	describeOperationMetadata := nm.DescribeOperationMetadata
	if describeOperationMetadata == nil {
		describeOperationMetadata = defaultDescribeOperationMetadata
	}
	var nondeterministicStepContext func(ctx context.Context, state interface{}, input interface{}, output interface{}) []interface{}
	switch {
	case nm.StepContext != nil:
		nondeterministicStepContext = nm.StepContext
	case nm.Step != nil:
		nondeterministicStepContext = func(ctx context.Context, state interface{}, input interface{}, output interface{}) []interface{} {
			return nm.Step(state, input, output)
		}
	default:
		panic("nondeterministic model must define Step or StepContext")
	}
	step := func(ctx context.Context, state, input, output interface{}) (bool, interface{}) {
		states := state.([]interface{})
		var allNextStates []interface{}
		for _, state := range states {
			if ctx.Err() != nil {
				return false, nil
			}
			allNextStates = append(allNextStates, nondeterministicStepContext(ctx, state, input, output)...)
		}
		if ctx.Err() != nil {
			return false, nil
		}
		uniqueNextStates := merge(allNextStates, equal)
		return len(uniqueNextStates) > 0, uniqueNextStates
	}
	var hashState func(state interface{}) uint64
	if nm.Hash != nil {
		// using XOR for order-independent combination of the
		// individual hashes, because state is a set
		hashState = func(state interface{}) uint64 {
			states := state.([]interface{})
			var h uint64
			for _, s := range states {
				h ^= nm.Hash(s)
			}
			return h
		}
	}
	backgroundContext := context.Background()
	return Model{
		Partition:      nm.Partition,
		PartitionEvent: nm.PartitionEvent,
		// we need this wrapper to convert a []interface{} to an interface{}
		Init: func() interface{} {
			return merge(nm.Init(), equal)
		},
		Step: func(state, input, output interface{}) (bool, interface{}) {
			return step(backgroundContext, state, input, output)
		},
		StepContext: step,
		// this operates on sets of states that have been merged, so we
		// don't need to check inclusion in both directions
		Equal: func(state1, state2 interface{}) bool {
			states1 := state1.([]interface{})
			states2 := state2.([]interface{})
			if len(states1) != len(states2) {
				return false
			}
			for _, s1 := range states1 {
				found := false
				for _, s2 := range states2 {
					if equal(s1, s2) {
						found = true
						break
					}
				}
				if !found {
					return false
				}
			}
			return true
		},
		Hash:              hashState,
		DescribeOperation: describeOperation,
		DescribeState: func(state interface{}) string {
			states := state.([]interface{})
			var descriptions []string
			for _, state := range states {
				descriptions = append(descriptions, describeState(state))
			}
			return fmt.Sprintf("{%s}", strings.Join(descriptions, ", "))
		},
		DescribeOperationMetadata: describeOperationMetadata,
	}
}

// noPartition is a fallback partition function that partitions the history
// into a single partition containing all of the operations.
func noPartition(history []Operation) [][]Operation {
	return [][]Operation{history}
}

// noPartitionEvent is a fallback partition function that partitions the
// history into a single partition containing all of the events.
func noPartitionEvent(history []Event) [][]Event {
	return [][]Event{history}
}

// shallowEqual is a fallback equality function that compares two states using
// ==.
func shallowEqual(state1, state2 interface{}) bool {
	return state1 == state2
}

// defaultDescribeOperation is a fallback to convert an operation to a string.
// It renders inputs and outputs using the "%v" format specifier.
func defaultDescribeOperation(input interface{}, output interface{}) string {
	return fmt.Sprintf("%v -> %v", input, output)
}

// defaultDescribeState is a fallback to convert a state to a string. It
// renders the state using the "%v" format specifier.
func defaultDescribeState(state interface{}) string {
	return fmt.Sprintf("%v", state)
}

// defaultDescribeOperationMetadata is a fallback to convert metadata to a
// string. It renders the metadata using the "%v" format specifier.
func defaultDescribeOperationMetadata(info interface{}) string {
	if info == nil {
		return ""
	}
	return fmt.Sprintf("%v", info)
}

// A CheckResult is the result of a linearizability check.
//
// Checking for linearizability is decidable, but it is an NP-hard problem, so
// the checker might take a long time. If a timeout is not given, functions in
// this package will always return Ok or Illegal, but if a timeout is supplied,
// then some functions may return Unknown. Depending on the use case, you can
// interpret an Unknown result as Ok (i.e., the tool didn't find a
// linearizability violation within the given timeout).
type CheckResult string

const (
	Unknown CheckResult = "Unknown" // timed out
	Ok      CheckResult = "Ok"
	Illegal CheckResult = "Illegal"
)
