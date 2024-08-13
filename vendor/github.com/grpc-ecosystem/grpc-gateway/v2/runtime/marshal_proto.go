package runtime

import (
	"errors"
	"io"

	"google.golang.org/protobuf/proto"
)

// ProtoMarshaller is a Marshaller which marshals/unmarshals into/from serialize proto bytes
type ProtoMarshaller struct{}

// ContentType always returns "application/octet-stream".
func (*ProtoMarshaller) ContentType(_ interface{}) string {
	return "application/octet-stream"
}

// Marshal marshals "value" into Proto
func (*ProtoMarshaller) Marshal(value interface{}) ([]byte, error) {
	message, ok := value.(proto.Message)
	if !ok {
		return nil, errors.New("unable to marshal non proto field")
	}
	return proto.Marshal(message)
}

// Unmarshal unmarshals proto "data" into "value"
func (*ProtoMarshaller) Unmarshal(data []byte, value interface{}) error {
	message, ok := value.(proto.Message)
	if !ok {
		return errors.New("unable to unmarshal non proto field")
	}
	return proto.Unmarshal(data, message)
}

// NewDecoder returns a Decoder which reads proto stream from "reader".
func (marshaller *ProtoMarshaller) NewDecoder(reader io.Reader) Decoder {
	return DecoderFunc(func(value interface{}) error {
		buffer, err := io.ReadAll(reader)
		if err != nil {
			return err
		}
		return marshaller.Unmarshal(buffer, value)
	})
}

// NewEncoder returns an Encoder which writes proto stream into "writer".
func (marshaller *ProtoMarshaller) NewEncoder(writer io.Writer) Encoder {
	return EncoderFunc(func(value interface{}) error {
		buffer, err := marshaller.Marshal(value)
		if err != nil {
			return err
		}
		if _, err := writer.Write(buffer); err != nil {
			return err
		}

		return nil
	})
}
