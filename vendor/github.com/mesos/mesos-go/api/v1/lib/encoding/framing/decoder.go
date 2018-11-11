package framing

type (
	// UnmarshalFunc translates bytes to objects
	UnmarshalFunc func([]byte, interface{}) error

	// Decoder reads and decodes Protobuf messages from an io.Reader.
	Decoder interface {
		// Decode reads the next encoded message from its input and stores it
		// in the value pointed to by m. If m isn't a proto.Message, Decode will panic.
		Decode(interface{}) error
	}

	// DecoderFunc is the functional adaptation of Decoder
	DecoderFunc func(interface{}) error
)

func (f DecoderFunc) Decode(m interface{}) error { return f(m) }

var _ = Decoder(DecoderFunc(nil))

// NewDecoder returns a new Decoder that reads from the given frame Reader.
func NewDecoder(r Reader, uf UnmarshalFunc) DecoderFunc {
	return func(m interface{}) error {
		// Note: the buf returned by ReadFrame will change over time, it can't be sub-sliced
		// and then those sub-slices retained. Examination of generated proto code seems to indicate
		// that byte buffers are copied vs. referenced by sub-slice (gogo protoc).
		frame, err := r.ReadFrame()
		if err != nil {
			return err
		}
		return uf(frame, m)
	}
}
