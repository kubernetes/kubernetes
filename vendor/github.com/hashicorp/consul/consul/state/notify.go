package state

import (
	"sync"
)

// NotifyGroup is used to allow a simple notification mechanism.
// Channels can be marked as waiting, and when notify is invoked,
// all the waiting channels get a message and are cleared from the
// notify list.
type NotifyGroup struct {
	l      sync.Mutex
	notify map[chan struct{}]struct{}
}

// Notify will do a non-blocking send to all waiting channels, and
// clear the notify list
func (n *NotifyGroup) Notify() {
	n.l.Lock()
	defer n.l.Unlock()
	for ch, _ := range n.notify {
		select {
		case ch <- struct{}{}:
		default:
		}
	}
	n.notify = nil
}

// Wait adds a channel to the notify group
func (n *NotifyGroup) Wait(ch chan struct{}) {
	n.l.Lock()
	defer n.l.Unlock()
	if n.notify == nil {
		n.notify = make(map[chan struct{}]struct{})
	}
	n.notify[ch] = struct{}{}
}

// Clear removes a channel from the notify group
func (n *NotifyGroup) Clear(ch chan struct{}) {
	n.l.Lock()
	defer n.l.Unlock()
	if n.notify == nil {
		return
	}
	delete(n.notify, ch)
}

// WaitCh allocates a channel that is subscribed to notifications
func (n *NotifyGroup) WaitCh() chan struct{} {
	ch := make(chan struct{}, 1)
	n.Wait(ch)
	return ch
}
