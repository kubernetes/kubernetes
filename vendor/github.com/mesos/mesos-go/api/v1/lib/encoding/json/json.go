package json

import (
	"encoding/json"

	"github.com/mesos/mesos-go/api/v1/lib/encoding"
	"github.com/mesos/mesos-go/api/v1/lib/encoding/framing"
)

// NewEncoder returns a new Encoder of Calls to JSON messages written to
// the given io.Writer.
func NewEncoder(s encoding.Sink) encoding.Encoder {
	w := s()
	return encoding.EncoderFunc(func(m encoding.Marshaler) error {
		b, err := json.Marshal(m)
		if err != nil {
			return err
		}
		return w.WriteFrame(b)
	})
}

// NewDecoder returns a new Decoder of JSON messages read from the given source.
func NewDecoder(s encoding.Source) encoding.Decoder {
	r := s()
	dec := framing.NewDecoder(r, json.Unmarshal)
	return encoding.DecoderFunc(func(u encoding.Unmarshaler) error { return dec.Decode(u) })
}
