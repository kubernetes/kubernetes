package tsdb

import (
	"container/heap"
	"encoding/binary"
	"sort"
	"strings"

	"github.com/influxdb/influxdb/influxql"
)

// EOF represents a "not found" key returned by a Cursor.
const EOF = int64(-1)

// Cursor represents an iterator over a series.
type Cursor interface {
	SeekTo(seek int64) (key int64, value interface{})
	Next() (key int64, value interface{})
	Ascending() bool
}

// MultiCursor returns a single cursor that combines the results of all cursors in order.
//
// If the same key is returned from multiple cursors then the first cursor
// specified will take precendence. A key will only be returned once from the
// returned cursor.
func MultiCursor(cursors ...Cursor) Cursor {
	return &multiCursor{
		cursors: cursors,
	}
}

// multiCursor represents a cursor that combines multiple cursors into one.
type multiCursor struct {
	cursors []Cursor
	heap    cursorHeap
	prev    int64 // previously read key
}

// Seek moves the cursor to a given key.
func (mc *multiCursor) SeekTo(seek int64) (int64, interface{}) {
	// Initialize heap.
	h := make(cursorHeap, 0, len(mc.cursors))
	for i, c := range mc.cursors {
		// Move cursor to position. Skip if it's empty.
		k, v := c.SeekTo(seek)
		if k == EOF {
			continue
		}

		// Append cursor to heap.
		h = append(h, &cursorHeapItem{
			key:      k,
			value:    v,
			cursor:   c,
			priority: len(mc.cursors) - i,
		})
	}

	heap.Init(&h)
	mc.heap = h
	mc.prev = EOF

	return mc.pop()
}

// Ascending returns the direction of the first cursor.
func (mc *multiCursor) Ascending() bool {
	if len(mc.cursors) == 0 {
		return true
	}
	return mc.cursors[0].Ascending()
}

// Next returns the next key/value from the cursor.
func (mc *multiCursor) Next() (int64, interface{}) { return mc.pop() }

// pop returns the next item from the heap.
// Reads the next key/value from item's cursor and puts it back on the heap.
func (mc *multiCursor) pop() (key int64, value interface{}) {
	// Read items until we have a key that doesn't match the previously read one.
	// This is to perform deduplication when there's multiple items with the same key.
	// The highest priority cursor will be read first and then remaining keys will be dropped.
	for {
		// Return EOF marker if there are no more items left.
		if len(mc.heap) == 0 {
			return EOF, nil
		}

		// Read the next item from the heap.
		item := heap.Pop(&mc.heap).(*cursorHeapItem)

		// Save the key/value for return.
		key, value = item.key, item.value

		// Read the next item from the cursor. Push back to heap if one exists.
		if item.key, item.value = item.cursor.Next(); item.key != EOF {
			heap.Push(&mc.heap, item)
		}

		// Skip if this key matches the previously returned one.
		if key == mc.prev {
			continue
		}

		mc.prev = key
		return
	}
}

// cursorHeap represents a heap of cursorHeapItems.
type cursorHeap []*cursorHeapItem

func (h cursorHeap) Len() int      { return len(h) }
func (h cursorHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h cursorHeap) Less(i, j int) bool {
	// Use priority if the keys are the same.
	if h[i].key == h[j].key {
		return h[i].priority > h[j].priority
	}

	// Otherwise compare based on cursor direction.
	if h[i].cursor.Ascending() {
		return h[i].key < h[j].key
	}
	return h[i].key > h[j].key
}

func (h *cursorHeap) Push(x interface{}) {
	*h = append(*h, x.(*cursorHeapItem))
}

func (h *cursorHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}

// cursorHeapItem is something we manage in a priority queue.
type cursorHeapItem struct {
	key      int64
	value    interface{}
	cursor   Cursor
	priority int
}

// TagSetCursor is virtual cursor that iterates over multiple TagsCursors.
type TagSetCursor struct {
	measurement   string            // Measurement name
	currentFields interface{}       // the current decoded and selected fields for the cursor in play
	tags          map[string]string // Tag key-value pairs
	cursors       []*TagsCursor     // Underlying tags cursors.
	currentTags   map[string]string // the current tags for the underlying series cursor in play

	SelectFields []string // fields to be selected

	// Min-heap of cursors ordered by timestamp.
	heap heap.Interface

	// Memoize the cursor's tagset-based key.
	memokey string
}

// NewTagSetCursor returns a instance of TagSetCursor.
func NewTagSetCursor(m string, t map[string]string, c []*TagsCursor, ascending bool) *TagSetCursor {
	return &TagSetCursor{
		measurement: m,
		tags:        t,
		cursors:     c,
		heap:        newPointHeap(ascending),
	}
}

func (tsc *TagSetCursor) key() string {
	if tsc.memokey == "" {
		if len(tsc.tags) == 0 {
			tsc.memokey = tsc.measurement
		} else {
			tsc.memokey = strings.Join([]string{tsc.measurement, string(MarshalTags(tsc.tags))}, "|")
		}
	}
	return tsc.memokey
}

func (tsc *TagSetCursor) Init(seek int64) {
	// Prime the buffers.
	for i := 0; i < len(tsc.cursors); i++ {
		k, v := tsc.cursors[i].SeekTo(seek)
		if k == EOF {
			k, v = tsc.cursors[i].Next()
		}
		if k == EOF {
			continue
		}

		heap.Push(tsc.heap, &pointHeapItem{
			timestamp: k,
			value:     v,
			cursor:    tsc.cursors[i],
		})
	}
}

// Next returns the next matching series-key, timestamp byte slice and meta tags for the tagset. Filtering
// is enforced on the values. If there is no matching value, then a nil result is returned.
func (tsc *TagSetCursor) Next(tmin, tmax int64) (int64, interface{}) {
	for {
		// If we're out of points, we're done.
		if tsc.heap.Len() == 0 {
			return -1, nil
		}

		// Grab the next point with the lowest timestamp.
		p := heap.Pop(tsc.heap).(*pointHeapItem)

		// We're done if the point is outside the query's time range [tmin:tmax).
		if p.timestamp != tmin && (p.timestamp < tmin || p.timestamp > tmax) {
			return -1, nil
		}

		// Save timestamp & value.
		timestamp, value := p.timestamp, p.value

		// Keep track of all fields for series cursor so we can
		// respond with them if asked
		tsc.currentFields = value

		// Keep track of the current tags for the series cursor so we can
		// respond with them if asked
		tsc.currentTags = p.cursor.tags

		// Advance the cursor.
		if nextKey, nextVal := p.cursor.Next(); nextKey != -1 {
			*p = pointHeapItem{
				timestamp: nextKey,
				value:     nextVal,
				cursor:    p.cursor,
			}
			heap.Push(tsc.heap, p)
		}

		// Value didn't match, look for the next one.
		if value == nil {
			continue
		}

		// Filter value.
		if p.cursor.filter != nil {
			// Convert value to a map for filter evaluation.
			m, ok := value.(map[string]interface{})
			if !ok {
				m = map[string]interface{}{tsc.SelectFields[0]: value}
			}

			// If filter fails then skip to the next value.
			if !influxql.EvalBool(p.cursor.filter, m) {
				continue
			}
		}

		// Filter out single field, if specified.
		if len(tsc.SelectFields) == 1 {
			if m, ok := value.(map[string]interface{}); ok {
				value = m[tsc.SelectFields[0]]
			}
			if value == nil {
				continue
			}
		}

		return timestamp, value
	}
}

// Fields returns the current fields of the current cursor
func (tsc *TagSetCursor) Fields() map[string]interface{} {
	switch v := tsc.currentFields.(type) {
	case map[string]interface{}:
		return v
	default:
		return map[string]interface{}{"": v}
	}
}

// Tags returns the current tags of the current cursor
// if there is no current currsor, it returns nil
func (tsc *TagSetCursor) Tags() map[string]string { return tsc.currentTags }

type pointHeapItem struct {
	timestamp int64
	value     interface{}
	cursor    *TagsCursor // cursor whence pointHeapItem came
}

type pointHeap []*pointHeapItem
type pointHeapReverse struct {
	pointHeap
}

func newPointHeap(ascending bool) heap.Interface {
	q := make(pointHeap, 0)
	heap.Init(&q)
	if ascending {
		return &q
	} else {
		return &pointHeapReverse{q}
	}
}

func (pq *pointHeapReverse) Less(i, j int) bool {
	return pq.pointHeap[i].timestamp > pq.pointHeap[j].timestamp
}

func (pq pointHeap) Len() int { return len(pq) }

func (pq pointHeap) Less(i, j int) bool {
	// We want a min-heap (points in chronological order), so use less than.
	return pq[i].timestamp < pq[j].timestamp
}

func (pq pointHeap) Swap(i, j int) { pq[i], pq[j] = pq[j], pq[i] }

func (pq *pointHeap) Push(x interface{}) {
	item := x.(*pointHeapItem)
	*pq = append(*pq, item)
}

func (pq *pointHeap) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

// TagsCursor is a cursor with attached tags and filter.
type TagsCursor struct {
	cursor Cursor
	filter influxql.Expr
	tags   map[string]string

	seek int64
	buf  struct {
		key   int64
		value interface{}
	}
}

// NewTagsCursor returns a new instance of a series cursor.
func NewTagsCursor(c Cursor, filter influxql.Expr, tags map[string]string) *TagsCursor {
	return &TagsCursor{
		cursor: c,
		filter: filter,
		tags:   tags,
		seek:   EOF,
	}
}

// Seek positions returning the key and value at that key.
func (c *TagsCursor) SeekTo(seek int64) (int64, interface{}) {
	// We've seeked on this cursor. This seek is after that previous cached seek
	// and the result it gave was after the key for this seek.
	//
	// In this case, any seek would just return what we got before, so there's
	// no point in reseeking.
	if c.seek != -1 && c.seek < seek && (c.buf.key == EOF || c.buf.key >= seek) {
		return c.buf.key, c.buf.value
	}

	// Seek to key/value in underlying cursor.
	key, value := c.cursor.SeekTo(seek)

	// Save the seek to the buffer.
	c.seek = seek
	c.buf.key, c.buf.value = key, value
	return key, value
}

// Next returns the next timestamp and value from the cursor.
func (c *TagsCursor) Next() (int64, interface{}) {
	// Invalidate the seek.
	c.seek = -1
	c.buf.key, c.buf.value = 0, nil

	// Return next key/value.
	return c.cursor.Next()
}

// TagSetCursors represents a sortable slice of TagSetCursors.
type TagSetCursors []*TagSetCursor

func (a TagSetCursors) Len() int           { return len(a) }
func (a TagSetCursors) Less(i, j int) bool { return a[i].key() < a[j].key() }
func (a TagSetCursors) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func (a TagSetCursors) Keys() []string {
	keys := []string{}
	for i := range a {
		keys = append(keys, a[i].key())
	}
	sort.Strings(keys)
	return keys
}

// btou64 converts an 8-byte slice into an uint64.
func btou64(b []byte) uint64 { return binary.BigEndian.Uint64(b) }
