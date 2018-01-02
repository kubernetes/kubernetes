package easyjson

import (
	"github.com/mailru/easyjson/jlexer"
	"github.com/mailru/easyjson/jwriter"
)

// RawMessage is a raw piece of JSON (number, string, bool, object, array or
// null) that is extracted without parsing and output as is during marshaling.
type RawMessage []byte

// MarshalEasyJSON does JSON marshaling using easyjson interface.
func (v *RawMessage) MarshalEasyJSON(w *jwriter.Writer) {
	if len(*v) == 0 {
		w.RawString("null")
	} else {
		w.Raw(*v, nil)
	}
}

// UnmarshalEasyJSON does JSON unmarshaling using easyjson interface.
func (v *RawMessage) UnmarshalEasyJSON(l *jlexer.Lexer) {
	*v = RawMessage(l.Raw())
}

// UnmarshalJSON implements encoding/json.Unmarshaler interface.
func (v *RawMessage) UnmarshalJSON(data []byte) error {
	*v = data
	return nil
}

var nullBytes = []byte("null")

// MarshalJSON implements encoding/json.Marshaler interface.
func (v RawMessage) MarshalJSON() ([]byte, error) {
	if len(v) == 0 {
		return nullBytes, nil
	}
	return v, nil
}

// IsDefined is required for integration with omitempty easyjson logic.
func (v *RawMessage) IsDefined() bool {
	return len(*v) > 0
}
