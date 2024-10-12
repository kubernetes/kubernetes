package queue

import (
	"errors"
	"sync"
)

var ErrQueueClosed = errors.New("the queue is closed for reading and writing")

// MessageQueue represents a threadsafe message queue to be used to retrieve or
// write messages to.
type MessageQueue struct {
	m        *sync.RWMutex
	c        *sync.Cond
	messages []interface{}
	closed   bool
}

// NewMessageQueue returns a new MessageQueue.
func NewMessageQueue() *MessageQueue {
	m := &sync.RWMutex{}
	return &MessageQueue{
		m:        m,
		c:        sync.NewCond(m),
		messages: []interface{}{},
	}
}

// Enqueue writes `msg` to the queue.
func (mq *MessageQueue) Enqueue(msg interface{}) error {
	mq.m.Lock()
	defer mq.m.Unlock()

	if mq.closed {
		return ErrQueueClosed
	}
	mq.messages = append(mq.messages, msg)
	// Signal a waiter that there is now a value available in the queue.
	mq.c.Signal()
	return nil
}

// Dequeue will read a value from the queue and remove it. If the queue
// is empty, this will block until the queue is closed or a value gets enqueued.
func (mq *MessageQueue) Dequeue() (interface{}, error) {
	mq.m.Lock()
	defer mq.m.Unlock()

	for !mq.closed && mq.size() == 0 {
		mq.c.Wait()
	}

	// We got woken up, check if it's because the queue got closed.
	if mq.closed {
		return nil, ErrQueueClosed
	}

	val := mq.messages[0]
	mq.messages[0] = nil
	mq.messages = mq.messages[1:]
	return val, nil
}

// Size returns the size of the queue.
func (mq *MessageQueue) Size() int {
	mq.m.RLock()
	defer mq.m.RUnlock()
	return mq.size()
}

// Nonexported size check to check if the queue is empty inside already locked functions.
func (mq *MessageQueue) size() int {
	return len(mq.messages)
}

// Close closes the queue for future writes or reads. Any attempts to read or write from the
// queue after close will return ErrQueueClosed. This is safe to call multiple times.
func (mq *MessageQueue) Close() {
	mq.m.Lock()
	defer mq.m.Unlock()

	// Already closed, noop
	if mq.closed {
		return
	}

	mq.messages = nil
	mq.closed = true
	// If there's anybody currently waiting on a value from Dequeue, we need to
	// broadcast so the read(s) can return ErrQueueClosed.
	mq.c.Broadcast()
}
