package debugger

import (
	"os"
	"os/signal"

	"syscall"
)

// compareSignal is the signal to trigger cache compare. For non-windows
// environment it's SIGUSR2.
var compareSignal = syscall.SIGUSR2

func (debugger *CacheDebugger) WaitForNotify(stopCh <-chan struct{}) {
	compareCh := make(chan os.Signal, 1)
	signal.Notify(compareCh, compareSignal)

	go func() {
		for {
			select {
			case <-stopCh:
				return
			case <-compareCh:
				debugger.Comparer.Compare()
				debugger.Dumper.DumpAll()
			}
		}
	}()
}
