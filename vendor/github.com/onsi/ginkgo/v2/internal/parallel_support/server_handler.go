package parallel_support

import (
	"io"
	"os"
	"sync"

	"github.com/onsi/ginkgo/v2/reporters"
	"github.com/onsi/ginkgo/v2/types"
)

type Void struct{}

var voidReceiver *Void = &Void{}
var voidSender Void

// ServerHandler is an RPC-compatible handler that is shared between the http server and the rpc server.
// It handles all the business logic to avoid duplication between the two servers

type ServerHandler struct {
	done              chan interface{}
	outputDestination io.Writer
	reporter          reporters.Reporter
	alives            []func() bool
	lock              *sync.Mutex
	beforeSuiteState  BeforeSuiteState
	parallelTotal     int
	counter           int
	counterLock       *sync.Mutex
	shouldAbort       bool

	numSuiteDidBegins int
	numSuiteDidEnds   int
	aggregatedReport  types.Report
	reportHoldingArea []types.SpecReport
}

func newServerHandler(parallelTotal int, reporter reporters.Reporter) *ServerHandler {
	return &ServerHandler{
		reporter:          reporter,
		lock:              &sync.Mutex{},
		counterLock:       &sync.Mutex{},
		alives:            make([]func() bool, parallelTotal),
		beforeSuiteState:  BeforeSuiteState{Data: nil, State: types.SpecStateInvalid},
		parallelTotal:     parallelTotal,
		outputDestination: os.Stdout,
		done:              make(chan interface{}),
	}
}

func (handler *ServerHandler) SpecSuiteWillBegin(report types.Report, _ *Void) error {
	handler.lock.Lock()
	defer handler.lock.Unlock()

	handler.numSuiteDidBegins += 1

	// all summaries are identical, so it's fine to simply emit the last one of these
	if handler.numSuiteDidBegins == handler.parallelTotal {
		handler.reporter.SuiteWillBegin(report)

		for _, summary := range handler.reportHoldingArea {
			handler.reporter.WillRun(summary)
			handler.reporter.DidRun(summary)
		}

		handler.reportHoldingArea = nil
	}

	return nil
}

func (handler *ServerHandler) DidRun(report types.SpecReport, _ *Void) error {
	handler.lock.Lock()
	defer handler.lock.Unlock()

	if handler.numSuiteDidBegins == handler.parallelTotal {
		handler.reporter.WillRun(report)
		handler.reporter.DidRun(report)
	} else {
		handler.reportHoldingArea = append(handler.reportHoldingArea, report)
	}

	return nil
}

func (handler *ServerHandler) SpecSuiteDidEnd(report types.Report, _ *Void) error {
	handler.lock.Lock()
	defer handler.lock.Unlock()

	handler.numSuiteDidEnds += 1
	if handler.numSuiteDidEnds == 1 {
		handler.aggregatedReport = report
	} else {
		handler.aggregatedReport = handler.aggregatedReport.Add(report)
	}

	if handler.numSuiteDidEnds == handler.parallelTotal {
		handler.reporter.SuiteDidEnd(handler.aggregatedReport)
		close(handler.done)
	}

	return nil
}

func (handler *ServerHandler) EmitOutput(output []byte, n *int) error {
	var err error
	*n, err = handler.outputDestination.Write(output)
	return err
}

func (handler *ServerHandler) registerAlive(proc int, alive func() bool) {
	handler.lock.Lock()
	defer handler.lock.Unlock()
	handler.alives[proc-1] = alive
}

func (handler *ServerHandler) procIsAlive(proc int) bool {
	handler.lock.Lock()
	defer handler.lock.Unlock()
	alive := handler.alives[proc-1]
	if alive == nil {
		return true
	}
	return alive()
}

func (handler *ServerHandler) haveNonprimaryProcsFinished() bool {
	for i := 2; i <= handler.parallelTotal; i++ {
		if handler.procIsAlive(i) {
			return false
		}
	}
	return true
}

func (handler *ServerHandler) BeforeSuiteCompleted(beforeSuiteState BeforeSuiteState, _ *Void) error {
	handler.lock.Lock()
	defer handler.lock.Unlock()
	handler.beforeSuiteState = beforeSuiteState

	return nil
}

func (handler *ServerHandler) BeforeSuiteState(_ Void, beforeSuiteState *BeforeSuiteState) error {
	proc1IsAlive := handler.procIsAlive(1)
	handler.lock.Lock()
	defer handler.lock.Unlock()
	if handler.beforeSuiteState.State == types.SpecStateInvalid {
		if proc1IsAlive {
			return ErrorEarly
		} else {
			return ErrorGone
		}
	}
	*beforeSuiteState = handler.beforeSuiteState
	return nil
}

func (handler *ServerHandler) HaveNonprimaryProcsFinished(_ Void, _ *Void) error {
	if handler.haveNonprimaryProcsFinished() {
		return nil
	} else {
		return ErrorEarly
	}
}

func (handler *ServerHandler) AggregatedNonprimaryProcsReport(_ Void, report *types.Report) error {
	if handler.haveNonprimaryProcsFinished() {
		handler.lock.Lock()
		defer handler.lock.Unlock()
		if handler.numSuiteDidEnds == handler.parallelTotal-1 {
			*report = handler.aggregatedReport
			return nil
		} else {
			return ErrorGone
		}
	} else {
		return ErrorEarly
	}
}

func (handler *ServerHandler) Counter(_ Void, counter *int) error {
	handler.counterLock.Lock()
	defer handler.counterLock.Unlock()
	*counter = handler.counter
	handler.counter++
	return nil
}

func (handler *ServerHandler) Abort(_ Void, _ *Void) error {
	handler.lock.Lock()
	defer handler.lock.Unlock()
	handler.shouldAbort = true
	return nil
}

func (handler *ServerHandler) ShouldAbort(_ Void, shouldAbort *bool) error {
	handler.lock.Lock()
	defer handler.lock.Unlock()
	*shouldAbort = handler.shouldAbort
	return nil
}
