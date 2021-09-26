package descriptor_test

import (
	"fmt"
	"testing"

	tpb "github.com/gogo/protobuf/proto/test_proto"
	"github.com/gogo/protobuf/protoc-gen-gogo/descriptor"
)

func TestMessage(t *testing.T) {
	var msg *descriptor.DescriptorProto
	fd, md := descriptor.ForMessage(msg)
	if pkg, want := fd.GetPackage(), "google.protobuf"; pkg != want {
		t.Errorf("descriptor.ForMessage(%T).GetPackage() = %q; want %q", msg, pkg, want)
	}
	if name, want := md.GetName(), "DescriptorProto"; name != want {
		t.Fatalf("descriptor.ForMessage(%T).GetName() = %q; want %q", msg, name, want)
	}
}

func Example_options() {
	var msg *tpb.MyMessageSet
	_, md := descriptor.ForMessage(msg)
	if md.GetOptions().GetMessageSetWireFormat() {
		fmt.Printf("%v uses option message_set_wire_format.\n", md.GetName())
	}

	// Output:
	// MyMessageSet uses option message_set_wire_format.
}
