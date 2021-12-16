package quic

import (
	"sync"

	"github.com/lucas-clemente/quic-go/internal/flowcontrol"
	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/wire"
)

type windowUpdateQueue struct {
	mutex sync.Mutex

	queue      map[protocol.StreamID]struct{} // used as a set
	queuedConn bool                           // connection-level window update

	streamGetter       streamGetter
	connFlowController flowcontrol.ConnectionFlowController
	callback           func(wire.Frame)
}

func newWindowUpdateQueue(
	streamGetter streamGetter,
	connFC flowcontrol.ConnectionFlowController,
	cb func(wire.Frame),
) *windowUpdateQueue {
	return &windowUpdateQueue{
		queue:              make(map[protocol.StreamID]struct{}),
		streamGetter:       streamGetter,
		connFlowController: connFC,
		callback:           cb,
	}
}

func (q *windowUpdateQueue) AddStream(id protocol.StreamID) {
	q.mutex.Lock()
	q.queue[id] = struct{}{}
	q.mutex.Unlock()
}

func (q *windowUpdateQueue) AddConnection() {
	q.mutex.Lock()
	q.queuedConn = true
	q.mutex.Unlock()
}

func (q *windowUpdateQueue) QueueAll() {
	q.mutex.Lock()
	// queue a connection-level window update
	if q.queuedConn {
		q.callback(&wire.MaxDataFrame{MaximumData: q.connFlowController.GetWindowUpdate()})
		q.queuedConn = false
	}
	// queue all stream-level window updates
	for id := range q.queue {
		delete(q.queue, id)
		str, err := q.streamGetter.GetOrOpenReceiveStream(id)
		if err != nil || str == nil { // the stream can be nil if it was completed before dequeing the window update
			continue
		}
		offset := str.getWindowUpdate()
		if offset == 0 { // can happen if we received a final offset, right after queueing the window update
			continue
		}
		q.callback(&wire.MaxStreamDataFrame{
			StreamID:          id,
			MaximumStreamData: offset,
		})
	}
	q.mutex.Unlock()
}
