package proto

import (
	"github.com/gogo/protobuf/proto"
	"github.com/mesos/mesos-go/api/v1/lib/encoding"
	"github.com/mesos/mesos-go/api/v1/lib/encoding/framing"
)

// NewEncoder returns a new Encoder of Calls to Protobuf messages written to
// the given io.Writer.
func NewEncoder(s encoding.Sink) encoding.Encoder {
	w := s()
	return encoding.EncoderFunc(func(m encoding.Marshaler) error {
		b, err := proto.Marshal(m.(proto.Message))
		if err != nil {
			return err
		}
		return w.WriteFrame(b)
	})
}

// NewDecoder returns a new Decoder of Protobuf messages read from the given Source.
func NewDecoder(s encoding.Source) encoding.Decoder {
	r := s()
	var (
		uf  = func(b []byte, m interface{}) error { return proto.Unmarshal(b, m.(proto.Message)) }
		dec = framing.NewDecoder(r, uf)
	)
	return encoding.DecoderFunc(func(u encoding.Unmarshaler) error { return dec.Decode(u) })
}
