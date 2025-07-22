package interrupt_handler

import (
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/onsi/ginkgo/v2/internal/parallel_support"
)

var ABORT_POLLING_INTERVAL = 500 * time.Millisecond

type InterruptCause uint

const (
	InterruptCauseInvalid InterruptCause = iota
	InterruptCauseSignal
	InterruptCauseAbortByOtherProcess
)

type InterruptLevel uint

const (
	InterruptLevelUninterrupted InterruptLevel = iota
	InterruptLevelCleanupAndReport
	InterruptLevelReportOnly
	InterruptLevelBailOut
)

func (ic InterruptCause) String() string {
	switch ic {
	case InterruptCauseSignal:
		return "Interrupted by User"
	case InterruptCauseAbortByOtherProcess:
		return "Interrupted by Other Ginkgo Process"
	}
	return "INVALID_INTERRUPT_CAUSE"
}

type InterruptStatus struct {
	Channel chan interface{}
	Level   InterruptLevel
	Cause   InterruptCause
}

func (s InterruptStatus) Interrupted() bool {
	return s.Level != InterruptLevelUninterrupted
}

func (s InterruptStatus) Message() string {
	return s.Cause.String()
}

func (s InterruptStatus) ShouldIncludeProgressReport() bool {
	return s.Cause != InterruptCauseAbortByOtherProcess
}

type InterruptHandlerInterface interface {
	Status() InterruptStatus
}

type InterruptHandler struct {
	c                 chan interface{}
	lock              *sync.Mutex
	level             InterruptLevel
	cause             InterruptCause
	client            parallel_support.Client
	stop              chan interface{}
	signals           []os.Signal
	requestAbortCheck chan interface{}
}

func NewInterruptHandler(client parallel_support.Client, signals ...os.Signal) *InterruptHandler {
	if len(signals) == 0 {
		signals = []os.Signal{os.Interrupt, syscall.SIGTERM}
	}
	handler := &InterruptHandler{
		c:                 make(chan interface{}),
		lock:              &sync.Mutex{},
		stop:              make(chan interface{}),
		requestAbortCheck: make(chan interface{}),
		client:            client,
		signals:           signals,
	}
	handler.registerForInterrupts()
	return handler
}

func (handler *InterruptHandler) Stop() {
	close(handler.stop)
}

func (handler *InterruptHandler) registerForInterrupts() {
	// os signal handling
	signalChannel := make(chan os.Signal, 1)
	signal.Notify(signalChannel, handler.signals...)

	// cross-process abort handling
	var abortChannel chan interface{}
	if handler.client != nil {
		abortChannel = make(chan interface{})
		go func() {
			pollTicker := time.NewTicker(ABORT_POLLING_INTERVAL)
			for {
				select {
				case <-pollTicker.C:
					if handler.client.ShouldAbort() {
						close(abortChannel)
						pollTicker.Stop()
						return
					}
				case <-handler.requestAbortCheck:
					if handler.client.ShouldAbort() {
						close(abortChannel)
						pollTicker.Stop()
						return
					}
				case <-handler.stop:
					pollTicker.Stop()
					return
				}
			}
		}()
	}

	go func(abortChannel chan interface{}) {
		var interruptCause InterruptCause
		for {
			select {
			case <-signalChannel:
				interruptCause = InterruptCauseSignal
			case <-abortChannel:
				interruptCause = InterruptCauseAbortByOtherProcess
			case <-handler.stop:
				signal.Stop(signalChannel)
				return
			}
			abortChannel = nil

			handler.lock.Lock()
			oldLevel := handler.level
			handler.cause = interruptCause
			if handler.level == InterruptLevelUninterrupted {
				handler.level = InterruptLevelCleanupAndReport
			} else if handler.level == InterruptLevelCleanupAndReport {
				handler.level = InterruptLevelReportOnly
			} else if handler.level == InterruptLevelReportOnly {
				handler.level = InterruptLevelBailOut
			}
			if handler.level != oldLevel {
				close(handler.c)
				handler.c = make(chan interface{})
			}
			handler.lock.Unlock()
		}
	}(abortChannel)
}

func (handler *InterruptHandler) Status() InterruptStatus {
	handler.lock.Lock()
	status := InterruptStatus{
		Level:   handler.level,
		Channel: handler.c,
		Cause:   handler.cause,
	}
	handler.lock.Unlock()

	if handler.client != nil && handler.client.ShouldAbort() && !status.Interrupted() {
		close(handler.requestAbortCheck)
		<-status.Channel
		return handler.Status()
	}

	return status
}
