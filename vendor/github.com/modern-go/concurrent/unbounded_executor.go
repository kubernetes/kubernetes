package concurrent

import (
	"context"
	"fmt"
	"reflect"
	"runtime"
	"runtime/debug"
	"sync"
	"time"
)

// HandlePanic logs goroutine panic by default
var HandlePanic = func(recovered interface{}, funcName string) {
	ErrorLogger.Println(fmt.Sprintf("%s panic: %v", funcName, recovered))
	ErrorLogger.Println(string(debug.Stack()))
}

// UnboundedExecutor is a executor without limits on counts of alive goroutines
// it tracks the goroutine started by it, and can cancel them when shutdown
type UnboundedExecutor struct {
	ctx                   context.Context
	cancel                context.CancelFunc
	activeGoroutinesMutex *sync.Mutex
	activeGoroutines      map[string]int
	HandlePanic           func(recovered interface{}, funcName string)
}

// GlobalUnboundedExecutor has the life cycle of the program itself
// any goroutine want to be shutdown before main exit can be started from this executor
// GlobalUnboundedExecutor expects the main function to call stop
// it does not magically knows the main function exits
var GlobalUnboundedExecutor = NewUnboundedExecutor()

// NewUnboundedExecutor creates a new UnboundedExecutor,
// UnboundedExecutor can not be created by &UnboundedExecutor{}
// HandlePanic can be set with a callback to override global HandlePanic
func NewUnboundedExecutor() *UnboundedExecutor {
	ctx, cancel := context.WithCancel(context.TODO())
	return &UnboundedExecutor{
		ctx:                   ctx,
		cancel:                cancel,
		activeGoroutinesMutex: &sync.Mutex{},
		activeGoroutines:      map[string]int{},
	}
}

// Go starts a new goroutine and tracks its lifecycle.
// Panic will be recovered and logged automatically, except for StopSignal
func (executor *UnboundedExecutor) Go(handler func(ctx context.Context)) {
	pc := reflect.ValueOf(handler).Pointer()
	f := runtime.FuncForPC(pc)
	funcName := f.Name()
	file, line := f.FileLine(pc)
	executor.activeGoroutinesMutex.Lock()
	defer executor.activeGoroutinesMutex.Unlock()
	startFrom := fmt.Sprintf("%s:%d", file, line)
	executor.activeGoroutines[startFrom] += 1
	go func() {
		defer func() {
			recovered := recover()
			// if you want to quit a goroutine without trigger HandlePanic
			// use runtime.Goexit() to quit
			if recovered != nil {
				if executor.HandlePanic == nil {
					HandlePanic(recovered, funcName)
				} else {
					executor.HandlePanic(recovered, funcName)
				}
			}
			executor.activeGoroutinesMutex.Lock()
			executor.activeGoroutines[startFrom] -= 1
			executor.activeGoroutinesMutex.Unlock()
		}()
		handler(executor.ctx)
	}()
}

// Stop cancel all goroutines started by this executor without wait
func (executor *UnboundedExecutor) Stop() {
	executor.cancel()
}

// StopAndWaitForever cancel all goroutines started by this executor and
// wait until all goroutines exited
func (executor *UnboundedExecutor) StopAndWaitForever() {
	executor.StopAndWait(context.Background())
}

// StopAndWait cancel all goroutines started by this executor and wait.
// Wait can be cancelled by the context passed in.
func (executor *UnboundedExecutor) StopAndWait(ctx context.Context) {
	executor.cancel()
	for {
		oneHundredMilliseconds := time.NewTimer(time.Millisecond * 100)
		select {
		case <-oneHundredMilliseconds.C:
			if executor.checkNoActiveGoroutines() {
				return
			}
		case <-ctx.Done():
			return
		}
	}
}

func (executor *UnboundedExecutor) checkNoActiveGoroutines() bool {
	executor.activeGoroutinesMutex.Lock()
	defer executor.activeGoroutinesMutex.Unlock()
	for startFrom, count := range executor.activeGoroutines {
		if count > 0 {
			InfoLogger.Println("UnboundedExecutor is still waiting goroutines to quit",
				"startFrom", startFrom,
				"count", count)
			return false
		}
	}
	return true
}
