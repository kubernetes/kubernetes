package tsm1

import (
	"math"

	"github.com/influxdata/influxdb/tsdb"
)

// multieFieldCursor wraps cursors for multiple fields on the same series
// key. Instead of returning a plain interface value in the call for Next(),
// it returns a map[string]interface{} for the field values
type multiFieldCursor struct {
	fields      []string
	cursors     []tsdb.Cursor
	ascending   bool
	keyBuffer   []int64
	valueBuffer []interface{}
}

// NewMultiFieldCursor returns an instance of Cursor that joins the results of cursors.
func NewMultiFieldCursor(fields []string, cursors []tsdb.Cursor, ascending bool) tsdb.Cursor {
	return &multiFieldCursor{
		fields:      fields,
		cursors:     cursors,
		ascending:   ascending,
		keyBuffer:   make([]int64, len(cursors)),
		valueBuffer: make([]interface{}, len(cursors)),
	}
}

func (m *multiFieldCursor) SeekTo(seek int64) (key int64, value interface{}) {
	for i, c := range m.cursors {
		m.keyBuffer[i], m.valueBuffer[i] = c.SeekTo(seek)
	}
	return m.read()
}

func (m *multiFieldCursor) Next() (int64, interface{}) {
	return m.read()
}

func (m *multiFieldCursor) Ascending() bool {
	return m.ascending
}

func (m *multiFieldCursor) read() (int64, interface{}) {
	t := int64(math.MaxInt64)
	if !m.ascending {
		t = int64(math.MinInt64)
	}

	// find the time we need to combine all fields
	for _, k := range m.keyBuffer {
		if k == tsdb.EOF {
			continue
		}
		if m.ascending && t > k {
			t = k
		} else if !m.ascending && t < k {
			t = k
		}
	}

	// get the value and advance each of the cursors that have the matching time
	if t == math.MinInt64 || t == math.MaxInt64 {
		return tsdb.EOF, nil
	}

	mm := make(map[string]interface{})
	for i, k := range m.keyBuffer {
		if k == t {
			mm[m.fields[i]] = m.valueBuffer[i]
			m.keyBuffer[i], m.valueBuffer[i] = m.cursors[i].Next()
		}
	}
	return t, mm
}
