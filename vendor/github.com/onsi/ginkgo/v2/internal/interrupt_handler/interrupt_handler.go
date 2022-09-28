package interrupt_handler

import (
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"syscall"
	"time"

	"github.com/onsi/ginkgo/v2/formatter"
	"github.com/onsi/ginkgo/v2/internal/parallel_support"
)

const TIMEOUT_REPEAT_INTERRUPT_MAXIMUM_DURATION = 30 * time.Second
const TIMEOUT_REPEAT_INTERRUPT_FRACTION_OF_TIMEOUT = 10
const ABORT_POLLING_INTERVAL = 500 * time.Millisecond
const ABORT_REPEAT_INTERRUPT_DURATION = 30 * time.Second

type InterruptCause uint

const (
	InterruptCauseInvalid InterruptCause = iota

	InterruptCauseSignal
	InterruptCauseTimeout
	InterruptCauseAbortByOtherProcess
)

func (ic InterruptCause) String() string {
	switch ic {
	case InterruptCauseSignal:
		return "Interrupted by User"
	case InterruptCauseTimeout:
		return "Interrupted by Timeout"
	case InterruptCauseAbortByOtherProcess:
		return "Interrupted by Other Ginkgo Process"
	}
	return "INVALID_INTERRUPT_CAUSE"
}

type InterruptStatus struct {
	Interrupted bool
	Channel     chan interface{}
	Cause       InterruptCause
}

type InterruptHandlerInterface interface {
	Status() InterruptStatus
	SetInterruptPlaceholderMessage(string)
	ClearInterruptPlaceholderMessage()
	InterruptMessageWithStackTraces() string
}

type InterruptHandler struct {
	c                           chan interface{}
	lock                        *sync.Mutex
	interrupted                 bool
	interruptPlaceholderMessage string
	interruptCause              InterruptCause
	client                      parallel_support.Client
	stop                        chan interface{}
}

func NewInterruptHandler(timeout time.Duration, client parallel_support.Client) *InterruptHandler {
	handler := &InterruptHandler{
		c:           make(chan interface{}),
		lock:        &sync.Mutex{},
		interrupted: false,
		stop:        make(chan interface{}),
		client:      client,
	}
	handler.registerForInterrupts(timeout)
	return handler
}

func (handler *InterruptHandler) Stop() {
	close(handler.stop)
}

func (handler *InterruptHandler) registerForInterrupts(timeout time.Duration) {
	// os signal handling
	signalChannel := make(chan os.Signal, 1)
	signal.Notify(signalChannel, os.Interrupt, syscall.SIGTERM)

	// timeout handling
	var timeoutChannel <-chan time.Time
	var timeoutTimer *time.Timer
	if timeout > 0 {
		timeoutTimer = time.NewTimer(timeout)
		timeoutChannel = timeoutTimer.C
	}

	// cross-process abort handling
	var abortChannel chan bool
	if handler.client != nil {
		abortChannel = make(chan bool)
		go func() {
			pollTicker := time.NewTicker(ABORT_POLLING_INTERVAL)
			for {
				select {
				case <-pollTicker.C:
					if handler.client.ShouldAbort() {
						abortChannel <- true
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

	// listen for any interrupt signals
	// note that some (timeouts, cross-process aborts) will only trigger once
	// for these we set up a ticker to keep interrupting the suite until it ends
	// this ensures any `AfterEach` or `AfterSuite`s that get stuck cleaning up
	// get interrupted eventually
	go func() {
		var interruptCause InterruptCause
		var repeatChannel <-chan time.Time
		var repeatTicker *time.Ticker
		for {
			select {
			case <-signalChannel:
				interruptCause = InterruptCauseSignal
			case <-timeoutChannel:
				interruptCause = InterruptCauseTimeout
				repeatInterruptTimeout := timeout / time.Duration(TIMEOUT_REPEAT_INTERRUPT_FRACTION_OF_TIMEOUT)
				if repeatInterruptTimeout > TIMEOUT_REPEAT_INTERRUPT_MAXIMUM_DURATION {
					repeatInterruptTimeout = TIMEOUT_REPEAT_INTERRUPT_MAXIMUM_DURATION
				}
				timeoutTimer.Stop()
				repeatTicker = time.NewTicker(repeatInterruptTimeout)
				repeatChannel = repeatTicker.C
			case <-abortChannel:
				interruptCause = InterruptCauseAbortByOtherProcess
				repeatTicker = time.NewTicker(ABORT_REPEAT_INTERRUPT_DURATION)
				repeatChannel = repeatTicker.C
			case <-repeatChannel:
				//do nothing, just interrupt again using the same interruptCause
			case <-handler.stop:
				if timeoutTimer != nil {
					timeoutTimer.Stop()
				}
				if repeatTicker != nil {
					repeatTicker.Stop()
				}
				signal.Stop(signalChannel)
				return
			}
			handler.lock.Lock()
			handler.interruptCause = interruptCause
			if handler.interruptPlaceholderMessage != "" {
				fmt.Println(handler.interruptPlaceholderMessage)
			}
			handler.interrupted = true
			close(handler.c)
			handler.c = make(chan interface{})
			handler.lock.Unlock()
		}
	}()
}

func (handler *InterruptHandler) Status() InterruptStatus {
	handler.lock.Lock()
	defer handler.lock.Unlock()

	return InterruptStatus{
		Interrupted: handler.interrupted,
		Channel:     handler.c,
		Cause:       handler.interruptCause,
	}
}

func (handler *InterruptHandler) SetInterruptPlaceholderMessage(message string) {
	handler.lock.Lock()
	defer handler.lock.Unlock()

	handler.interruptPlaceholderMessage = message
}

func (handler *InterruptHandler) ClearInterruptPlaceholderMessage() {
	handler.lock.Lock()
	defer handler.lock.Unlock()

	handler.interruptPlaceholderMessage = ""
}

func (handler *InterruptHandler) InterruptMessageWithStackTraces() string {
	handler.lock.Lock()
	out := fmt.Sprintf("%s\n\n", handler.interruptCause.String())
	defer handler.lock.Unlock()
	if handler.interruptCause == InterruptCauseAbortByOtherProcess {
		return out
	}
	out += "Here's a stack trace of all running goroutines:\n"
	buf := make([]byte, 8192)
	for {
		n := runtime.Stack(buf, true)
		if n < len(buf) {
			buf = buf[:n]
			break
		}
		buf = make([]byte, 2*len(buf))
	}
	out += formatter.Fi(1, "%s", string(buf))
	return out
}
