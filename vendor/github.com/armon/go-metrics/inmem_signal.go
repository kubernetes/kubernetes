package metrics

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"os/signal"
	"sync"
	"syscall"
)

// InmemSignal is used to listen for a given signal, and when received,
// to dump the current metrics from the InmemSink to an io.Writer
type InmemSignal struct {
	signal syscall.Signal
	inm    *InmemSink
	w      io.Writer
	sigCh  chan os.Signal

	stop     bool
	stopCh   chan struct{}
	stopLock sync.Mutex
}

// NewInmemSignal creates a new InmemSignal which listens for a given signal,
// and dumps the current metrics out to a writer
func NewInmemSignal(inmem *InmemSink, sig syscall.Signal, w io.Writer) *InmemSignal {
	i := &InmemSignal{
		signal: sig,
		inm:    inmem,
		w:      w,
		sigCh:  make(chan os.Signal, 1),
		stopCh: make(chan struct{}),
	}
	signal.Notify(i.sigCh, sig)
	go i.run()
	return i
}

// DefaultInmemSignal returns a new InmemSignal that responds to SIGUSR1
// and writes output to stderr. Windows uses SIGBREAK
func DefaultInmemSignal(inmem *InmemSink) *InmemSignal {
	return NewInmemSignal(inmem, DefaultSignal, os.Stderr)
}

// Stop is used to stop the InmemSignal from listening
func (i *InmemSignal) Stop() {
	i.stopLock.Lock()
	defer i.stopLock.Unlock()

	if i.stop {
		return
	}
	i.stop = true
	close(i.stopCh)
	signal.Stop(i.sigCh)
}

// run is a long running routine that handles signals
func (i *InmemSignal) run() {
	for {
		select {
		case <-i.sigCh:
			i.dumpStats()
		case <-i.stopCh:
			return
		}
	}
}

// dumpStats is used to dump the data to output writer
func (i *InmemSignal) dumpStats() {
	buf := bytes.NewBuffer(nil)

	data := i.inm.Data()
	// Skip the last period which is still being aggregated
	for i := 0; i < len(data)-1; i++ {
		intv := data[i]
		intv.RLock()
		for name, val := range intv.Gauges {
			fmt.Fprintf(buf, "[%v][G] '%s': %0.3f\n", intv.Interval, name, val)
		}
		for name, vals := range intv.Points {
			for _, val := range vals {
				fmt.Fprintf(buf, "[%v][P] '%s': %0.3f\n", intv.Interval, name, val)
			}
		}
		for name, agg := range intv.Counters {
			fmt.Fprintf(buf, "[%v][C] '%s': %s\n", intv.Interval, name, agg)
		}
		for name, agg := range intv.Samples {
			fmt.Fprintf(buf, "[%v][S] '%s': %s\n", intv.Interval, name, agg)
		}
		intv.RUnlock()
	}

	// Write out the bytes
	i.w.Write(buf.Bytes())
}
