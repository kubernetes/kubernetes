package debugger

import (
	"os"
	"os/signal"
)

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
