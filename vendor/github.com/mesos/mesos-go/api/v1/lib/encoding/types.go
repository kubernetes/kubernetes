package encoding

import (
	"encoding/json"
	"io"

	pb "github.com/gogo/protobuf/proto"
	"github.com/mesos/mesos-go/api/v1/lib/encoding/framing"
)

type MediaType string

// ContentType returns the HTTP Content-Type associated with the MediaType
func (m MediaType) ContentType() string { return string(m) }

type (
	Source func() framing.Reader
	Sink   func() framing.Writer

	// A Codec composes encoding and decoding of a serialization format.
	Codec struct {
		Name       string
		Type       MediaType
		NewEncoder func(Sink) Encoder
		NewDecoder func(Source) Decoder
	}

	SourceFactory interface {
		NewSource(r io.Reader) Source
	}
	SourceFactoryFunc func(r io.Reader) Source

	SinkFactory interface {
		NewSink(w io.Writer) Sink
	}
	SinkFactoryFunc func(w io.Writer) Sink
)

func (f SourceFactoryFunc) NewSource(r io.Reader) Source { return f(r) }

func (f SinkFactoryFunc) NewSink(w io.Writer) Sink { return f(w) }

var (
	_ = SourceFactory(SourceFactoryFunc(nil))
	_ = SinkFactory(SinkFactoryFunc(nil))
)

// SourceReader returns a Source that buffers all input from the given io.Reader
// and returns the contents in a single frame.
func SourceReader(r io.Reader) Source {
	ch := make(chan framing.ReaderFunc, 1)
	ch <- framing.ReadAll(r)
	return func() framing.Reader {
		select {
		case f := <-ch:
			return f
		default:
			return framing.ReaderFunc(framing.EOFReaderFunc)
		}
	}
}

// SinkWriter returns a Sink that sends a frame to an io.Writer with no decoration.
func SinkWriter(w io.Writer) Sink { return func() framing.Writer { return framing.WriterFor(w) } }

// String implements the fmt.Stringer interface.
func (c *Codec) String() string {
	if c == nil {
		return ""
	}
	return c.Name
}

type (
	// Marshaler composes the supported marshaling formats.
	Marshaler interface {
		pb.Marshaler
		json.Marshaler
	}
	// Unmarshaler composes the supporter unmarshaling formats.
	Unmarshaler interface {
		pb.Unmarshaler
		json.Unmarshaler
	}
	// An Encoder encodes a given Marshaler or returns an error in case of failure.
	Encoder interface {
		Encode(Marshaler) error
	}

	// EncoderFunc is the functional adapter for Encoder
	EncoderFunc func(Marshaler) error

	// A Decoder decodes a given Unmarshaler or returns an error in case of failure.
	Decoder interface {
		Decode(Unmarshaler) error
	}

	// DecoderFunc is the functional adapter for Decoder
	DecoderFunc func(Unmarshaler) error
)

// Decode implements the Decoder interface
func (f DecoderFunc) Decode(u Unmarshaler) error { return f(u) }

// Encode implements the Encoder interface
func (f EncoderFunc) Encode(m Marshaler) error { return f(m) }

var (
	_ = Encoder(EncoderFunc(nil))
	_ = Decoder(DecoderFunc(nil))
)
