package dbus

import (
	"sync"
)

// NewSequentialSignalHandler returns an instance of a new
// signal handler that guarantees sequential processing of signals. It is a
// guarantee of this signal handler that signals will be written to
// channels in the order they are received on the DBus connection.
func NewSequentialSignalHandler() SignalHandler {
	return &sequentialSignalHandler{}
}

type sequentialSignalHandler struct {
	mu      sync.RWMutex
	closed  bool
	signals []*sequentialSignalChannelData
}

func (sh *sequentialSignalHandler) DeliverSignal(intf, name string, signal *Signal) {
	sh.mu.RLock()
	defer sh.mu.RUnlock()
	if sh.closed {
		return
	}
	for _, scd := range sh.signals {
		scd.deliver(signal)
	}
}

func (sh *sequentialSignalHandler) Terminate() {
	sh.mu.Lock()
	defer sh.mu.Unlock()
	if sh.closed {
		return
	}

	for _, scd := range sh.signals {
		scd.close()
		close(scd.ch)
	}
	sh.closed = true
	sh.signals = nil
}

func (sh *sequentialSignalHandler) AddSignal(ch chan<- *Signal) {
	sh.mu.Lock()
	defer sh.mu.Unlock()
	if sh.closed {
		return
	}
	sh.signals = append(sh.signals, newSequentialSignalChannelData(ch))
}

func (sh *sequentialSignalHandler) RemoveSignal(ch chan<- *Signal) {
	sh.mu.Lock()
	defer sh.mu.Unlock()
	if sh.closed {
		return
	}
	for i := len(sh.signals) - 1; i >= 0; i-- {
		if ch == sh.signals[i].ch {
			sh.signals[i].close()
			copy(sh.signals[i:], sh.signals[i+1:])
			sh.signals[len(sh.signals)-1] = nil
			sh.signals = sh.signals[:len(sh.signals)-1]
		}
	}
}

type sequentialSignalChannelData struct {
	ch   chan<- *Signal
	in   chan *Signal
	done chan struct{}
}

func newSequentialSignalChannelData(ch chan<- *Signal) *sequentialSignalChannelData {
	scd := &sequentialSignalChannelData{
		ch:   ch,
		in:   make(chan *Signal),
		done: make(chan struct{}),
	}
	go scd.bufferSignals()
	return scd
}

func (scd *sequentialSignalChannelData) bufferSignals() {
	defer close(scd.done)

	// Ensure that signals are delivered to scd.ch in the same
	// order they are received from scd.in.
	var queue []*Signal
	for {
		if len(queue) == 0 {
			signal, ok := <- scd.in
			if !ok {
				return
			}
			queue = append(queue, signal)
		}
		select {
		case scd.ch <- queue[0]:
			copy(queue, queue[1:])
			queue[len(queue)-1] = nil
			queue = queue[:len(queue)-1]
		case signal, ok := <-scd.in:
			if !ok {
				return
			}
			queue = append(queue, signal)
		}
	}
}

func (scd *sequentialSignalChannelData) deliver(signal *Signal) {
	scd.in <- signal
}

func (scd *sequentialSignalChannelData) close() {
	close(scd.in)
	// Ensure that bufferSignals() has exited and won't attempt
	// any future sends on scd.ch
	<-scd.done
}
