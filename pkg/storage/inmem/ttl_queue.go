package inmem

import (
	"fmt"
	"time"
	"sync"
	"container/heap"
	"math"
)

// expiryManager manages time based expirations.
// An interesting part of the design is that once scheduled items are never removed;
// instead once we see an expired key, we then trigger a check to see if it has actually expired.
type expiryManager struct {
	mutex          sync.Mutex
	queue          expiryQueue
	store          *store

	minChanged     bool
	minChangedChan chan bool
}

type expiringItem struct {
	bucket *bucket
	key    string
	expiry uint64
}

func newExpiryManager(store *store) *expiryManager {
	m := &expiryManager{
		store: store,
	}
	return m
}

// A expiryQueue implements heap.Interface and holds expiringItems,
// so we can efficiently pull of the next item
type expiryQueue []expiringItem

var _ heap.Interface = &expiryQueue{}

func (q expiryQueue) Len() int {
	return len(q)
}

func (q expiryQueue) Less(i, j int) bool {
	return q[i].expiry > q[j].expiry
}

func (q expiryQueue) Swap(i, j int) {
	q[i], q[j] = q[j], q[i]
}

func (q *expiryQueue) Push(x interface{}) {
	item := x.(expiringItem)
	*q = append(*q, item)
}

func (q *expiryQueue) Pop() interface{} {
	old := *q
	n := len(old)
	item := old[n - 1]
	*q = old[0 : n - 1]
	return item
}

func (m *expiryManager) add(bucket *bucket, key string, expiry uint64) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	var oldMin uint64
	if len(m.queue) != 0 {
		oldMin = m.queue[0].expiry
	}
	heap.Push(&m.queue, expiringItem{ bucket,  key,  expiry})
	newMin := m.queue[0].expiry
	if !m.minChanged && oldMin != newMin {
		m.minChanged = true
		m.minChangedChan <- true
	}
}

func (m *expiryManager) runOnce() {
	var paths []expiringItem

	m.mutex.Lock()
	now := uint64(time.Now().Unix())
	for {
		if len(m.queue) == 0 {
			break
		}
		nextExpiry := m.queue[0].expiry
		if nextExpiry > now {
			break
		}
		paths = append(paths, m.queue[0])
		heap.Pop(&m.queue)
	}
	m.minChanged = false
	m.mutex.Unlock()

	if len(paths) != 0 {
		err := m.store.checkExpiration(paths)
		if err != nil {
			// TODO: log, sleep, reattempt current batch?
			panic(fmt.Errorf("error processing expiration: %v", err))
		}
	}
}

func (m*expiryManager) wait() {
	var delaySeconds uint64

	m.mutex.Lock()
	now := uint64(time.Now().Unix())
	if len(m.queue) == 0 {
		delaySeconds = math.MaxInt32
	} else if now > m.queue[0].expiry {
		delaySeconds = now - m.queue[0].expiry
	}
	m.mutex.Unlock()

	if delaySeconds <= 0 {
		return
	}

	timerChan := time.NewTimer(time.Second * time.Duration(delaySeconds)).C
	select {
	case <-timerChan:
	case <-m.minChangedChan:
	}
}

func (m*expiryManager) Run() {
	for {
		m.wait()
		m.runOnce()
	}
}