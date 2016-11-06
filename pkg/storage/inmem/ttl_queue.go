package inmem

import (
	"time"
	"sync"
	"container/heap"
	"math"
)

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
	expiry int64
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

func (m *expiryManager) add(path string, expiry int64) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	oldMin := math.MinInt64
	if len(m.queue) != 0 {
		oldMin = m.queue[0].expiry
	}
	heap.Push(m.queue, expiringItem{path, expiry})
	newMin := m.queue[0]
	if !m.minChanged && oldMin != newMin {
		m.minChanged = true
		m.minChanged <- true
	}
}

func (m *expiryManager) runOnce() {
	var paths []expiringItem

	m.mutex.Lock()
	now := time.Now().Unix()
	for {
		if len(m.queue) == 0 {
			break
		}
		nextExpiry := m.queue[0].expiry
		if nextExpiry > now {
			break
		}
		paths = append(paths, m.queue[0])
		heap.Pop(m.queue)
	}
	m.minChanged = false
	m.mutex.Unlock()

	if len(paths) != 0 {
		m.store.checkExpiration(paths)
	}
}

func (m*expiryManager) wait() {
	var delaySeconds int32

	m.mutex.Lock()
	now := time.Now().Unix()
	if len(m.queue) == 0 {
		delaySeconds = math.MaxInt32
	} else {
		delaySeconds = now - m.queue[0].expiry
	}
	m.mutex.Unlock()

	if delaySeconds <= 0 {
		return
	}

	timerChan := time.NewTimer(time.Second * delaySeconds).C
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