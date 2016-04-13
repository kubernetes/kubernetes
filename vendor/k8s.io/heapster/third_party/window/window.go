// source: https://github.com/jbrukh/window
package window

// An array-based moving window; a moving window
// is a queue with a maximum size and the property
// that when the size is reached, pushing a new
// element into the queue causes the head to be
// popped.
//
// You can optimize the amount of slice copying
// that the MovingWindow will be doing by trading
// off with space complexity. Namely, the underlying
// array is allocated with a size that is the
// multiple of the intended capacity of the queue
// so that copying is less frequent.
type MovingWindow struct {
	arr  []interface{}
	size int
	head int
	tail int
}

// PushBack will push a new piece of data into
// the moving window
func (m *MovingWindow) PushBack(v interface{}) {
	// if the array is full, rewind
	if m.tail == len(m.arr) {
		m.rewind()
	}
	// push the value
	m.arr[m.tail] = v
	// check if the window is full,
	// and move head pointer appropriately
	if m.tail-m.head >= m.size {
		m.head++
	}
	m.tail++
}

// rewind will copy the last size-1 elements
// from the end of the underlying array to
// the front, starting at index 0.
func (m *MovingWindow) rewind() {
	l := len(m.arr)
	for i := 0; i < m.size-1; i++ {
		m.arr[i] = m.arr[l-m.size+i+1]
	}
	m.head, m.tail = 0, m.size-1
}

// Slice will present the MovingWindow in
// the form of a slice. This operation never
// requires array copying of any kind.
//
// Note that this value is guaranteed to be
// good only until the next call to push. If
// you wish to save the reference, you should
// make a copy.
func (m *MovingWindow) Slice() []interface{} {
	return m.arr[m.head:m.tail]
}

// Size returns the size of the moving window,
// which is set at initialization
func (m *MovingWindow) Size() int {
	return m.size
}

// New creates a new moving window,
// with the size and multiple specified.
//
// This data structures trades off space
// and copying complexity; more precisely,
// the number of moving windows that can
// be displayed without having to do any
// array copying is proportional to approx 1/M,
// where M is the multiple.
func New(size, multiple int) *MovingWindow {
	if size < 1 || multiple < 1 {
		panic("Must have positive size and multiple")
	}
	capacity := size * multiple
	return &MovingWindow{
		arr:  make([]interface{}, capacity, capacity),
		size: size,
	}
}
