package codecs

import (
	"github.com/mesos/mesos-go/api/v1/lib/encoding"
	"github.com/mesos/mesos-go/api/v1/lib/encoding/json"
	"github.com/mesos/mesos-go/api/v1/lib/encoding/proto"
)

const (
	// MediaTypeProtobuf is the Protobuf serialization format media type.
	MediaTypeProtobuf = encoding.MediaType("application/x-protobuf")
	// MediaTypeJSON is the JSON serialiation format media type.
	MediaTypeJSON = encoding.MediaType("application/json")

	NameProtobuf = "protobuf"
	NameJSON     = "json"
)

// ByMediaType are pre-configured default Codecs, ready to use OOTB
var ByMediaType = map[encoding.MediaType]encoding.Codec{
	MediaTypeProtobuf: encoding.Codec{
		Name:       NameProtobuf,
		Type:       MediaTypeProtobuf,
		NewEncoder: proto.NewEncoder,
		NewDecoder: proto.NewDecoder,
	},
	MediaTypeJSON: encoding.Codec{
		Name:       NameJSON,
		Type:       MediaTypeJSON,
		NewEncoder: json.NewEncoder,
		NewDecoder: json.NewDecoder,
	},
}
