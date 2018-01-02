package cache

import "gopkg.in/src-d/go-git.v4/plumbing"

// queue is a basic FIFO queue based on a circular list that resize as needed.
type queue struct {
	elements []plumbing.Hash
	size     int
	head     int
	tail     int
	count    int
}

// newQueue returns a queue with the specified initial size
func newQueue(size int) *queue {
	return &queue{
		elements: make([]plumbing.Hash, size),
		size:     size,
	}
}

// Push adds a node to the queue.
func (q *queue) Push(h plumbing.Hash) {
	if q.head == q.tail && q.count > 0 {
		elements := make([]plumbing.Hash, len(q.elements)+q.size)
		copy(elements, q.elements[q.head:])
		copy(elements[len(q.elements)-q.head:], q.elements[:q.head])
		q.head = 0
		q.tail = len(q.elements)
		q.elements = elements
	}
	q.elements[q.tail] = h
	q.tail = (q.tail + 1) % len(q.elements)
	q.count++
}

// Pop removes and returns a Hash from the queue in first to last order.
func (q *queue) Pop() plumbing.Hash {
	if q.count == 0 {
		return plumbing.ZeroHash
	}
	node := q.elements[q.head]
	q.head = (q.head + 1) % len(q.elements)
	q.count--
	return node
}
