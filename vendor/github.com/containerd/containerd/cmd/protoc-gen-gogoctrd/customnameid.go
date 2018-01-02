package main

import (
	"strings"

	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/proto"
	"github.com/gogo/protobuf/protoc-gen-gogo/descriptor"
	"github.com/gogo/protobuf/protoc-gen-gogo/generator"
	"github.com/gogo/protobuf/vanity"
)

// CustomNameID preprocess the field, and set the [(gogoproto.customname) = "..."]
// if necessary, in order to avoid setting `gogoproto.customname` manually.
// The automatically assigned name should conform to Golang convention.
func CustomNameID(file *descriptor.FileDescriptorProto) {

	f := func(field *descriptor.FieldDescriptorProto) {
		// Skip if [(gogoproto.customname) = "..."] has already been set.
		if gogoproto.IsCustomName(field) {
			return
		}
		// Skip if embedded
		if gogoproto.IsEmbed(field) {
			return
		}
		if field.OneofIndex != nil {
			return
		}
		fieldName := generator.CamelCase(*field.Name)
		switch {
		case *field.Name == "id":
			// id -> ID
			fieldName = "ID"
		case strings.HasPrefix(*field.Name, "id_"):
			// id_some -> IDSome
			fieldName = "ID" + fieldName[2:]
		case strings.HasSuffix(*field.Name, "_id"):
			// some_id -> SomeID
			fieldName = fieldName[:len(fieldName)-2] + "ID"
		case strings.HasSuffix(*field.Name, "_ids"):
			// some_ids -> SomeIDs
			fieldName = fieldName[:len(fieldName)-3] + "IDs"
		default:
			return
		}
		if field.Options == nil {
			field.Options = &descriptor.FieldOptions{}
		}
		if err := proto.SetExtension(field.Options, gogoproto.E_Customname, &fieldName); err != nil {
			panic(err)
		}
	}

	// Iterate through all fields in file
	vanity.ForEachFieldExcludingExtensions(file.MessageType, f)
}
