package tsdb

import (
	"encoding/json"
)

// Mapper is the interface all Mapper types must implement.
type Mapper interface {
	Open() error
	TagSets() []string
	Fields() []string
	NextChunk() (interface{}, error)
	Close()
}

// StatefulMapper encapsulates a Mapper and some state that the executor needs to
// track for that mapper.
type StatefulMapper struct {
	Mapper
	bufferedChunk *MapperOutput // Last read chunk.
	drained       bool
}

// NextChunk wraps a RawMapper and some state.
func (sm *StatefulMapper) NextChunk() (*MapperOutput, error) {
	c, err := sm.Mapper.NextChunk()
	if err != nil {
		return nil, err
	}
	chunk, ok := c.(*MapperOutput)
	if !ok {
		if chunk == interface{}(nil) {
			return nil, nil
		}
	}
	return chunk, nil
}

// MapperValue is a complex type, which can encapsulate data from both raw and aggregate
// mappers. This currently allows marshalling and network system to remain simpler. For
// aggregate output Time is ignored, and actual Time-Value pairs are contained soley
// within the Value field.
type MapperValue struct {
	Time  int64             `json:"time,omitempty"`  // Ignored for aggregate output.
	Value interface{}       `json:"value,omitempty"` // For aggregate, contains interval time multiple values.
	Tags  map[string]string `json:"tags,omitempty"`  // Meta tags for results
}

// MapperValueJSON is the JSON-encoded representation of MapperValue. Because MapperValue is
// a complex type, custom JSON encoding is required so that none of the types contained within
// a MapperValue are "lost", and so the data are encoded as byte slices where necessary.
type MapperValueJSON struct {
	Time    int64             `json:"time,omitempty"`
	RawData []byte            `json:"rdata,omitempty"`
	AggData [][]byte          `json:"adata,omitempty"`
	Tags    map[string]string `json:"tags,omitempty"`
}

// MarshalJSON returns the JSON-encoded representation of a MapperValue.
func (mv *MapperValue) MarshalJSON() ([]byte, error) {
	o := &MapperValueJSON{
		Time:    mv.Time,
		AggData: make([][]byte, 0),
		Tags:    mv.Tags,
	}

	o.Time = mv.Time
	o.Tags = mv.Tags
	if values, ok := mv.Value.([]interface{}); ok {
		// Value contain a slice of more values. This happens only with
		// aggregate output.
		for _, v := range values {
			b, err := json.Marshal(v)
			if err != nil {
				return nil, err
			}
			o.AggData = append(o.AggData, b)
		}
	} else {
		// If must be raw output, so just marshal the single value.
		b, err := json.Marshal(mv.Value)
		if err != nil {
			return nil, err
		}
		o.RawData = b
	}
	return json.Marshal(o)
}

type MapperValues []*MapperValue

func (a MapperValues) Len() int           { return len(a) }
func (a MapperValues) Less(i, j int) bool { return a[i].Time < a[j].Time }
func (a MapperValues) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

type MapperOutput struct {
	Name      string            `json:"name,omitempty"`
	Tags      map[string]string `json:"tags,omitempty"`
	Fields    []string          `json:"fields,omitempty"`    // Field names of returned data.
	Values    []*MapperValue    `json:"values,omitempty"`    // For aggregates contains a single value at [0]
	CursorKey string            `json:"cursorkey,omitempty"` // Tagset-based key for the source cursor. Cached for performance reasons.
}

// MapperOutputJSON is the JSON-encoded representation of MapperOutput. The query data is represented
// as a raw JSON message, so decode is delayed, and can proceed in a custom manner.
type MapperOutputJSON struct {
	Name      string            `json:"name,omitempty"`
	Tags      map[string]string `json:"tags,omitempty"`
	Fields    []string          `json:"fields,omitempty"`    // Field names of returned data.
	CursorKey string            `json:"cursorkey,omitempty"` // Tagset-based key for the source cursor.
	Values    json.RawMessage   `json:"values,omitempty"`
}

// MarshalJSON returns the JSON-encoded representation of a MapperOutput.
func (mo *MapperOutput) MarshalJSON() ([]byte, error) {
	o := &MapperOutputJSON{
		Name:      mo.Name,
		Tags:      mo.Tags,
		Fields:    mo.Fields,
		CursorKey: mo.CursorKey,
	}
	data, err := json.Marshal(mo.Values)
	if err != nil {
		return nil, err
	}
	o.Values = data

	return json.Marshal(o)
}

func (mo *MapperOutput) key() string {
	return mo.CursorKey
}

// uniqueStrings returns a slice of unique strings from all lists in a.
func uniqueStrings(a ...[]string) []string {
	// Calculate unique set of strings.
	m := make(map[string]struct{})
	for _, strs := range a {
		for _, str := range strs {
			m[str] = struct{}{}
		}
	}

	// Convert back to slice.
	result := make([]string, 0, len(m))
	for k := range m {
		result = append(result, k)
	}
	return result
}
