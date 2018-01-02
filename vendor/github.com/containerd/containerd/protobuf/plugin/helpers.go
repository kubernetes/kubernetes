package plugin

import (
	"github.com/gogo/protobuf/proto"
	"github.com/gogo/protobuf/protoc-gen-gogo/descriptor"
)

// FieldpathEnabled returns true if E_Fieldpath is enabled
func FieldpathEnabled(file *descriptor.FileDescriptorProto, message *descriptor.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_Fieldpath, proto.GetBoolExtension(file.Options, E_FieldpathAll, false))
}
