package runtime

import (
	"encoding/json"
	"io"
	"strings"

	descriptor2 "github.com/golang/protobuf/descriptor"
	"github.com/golang/protobuf/protoc-gen-go/descriptor"
	"google.golang.org/genproto/protobuf/field_mask"
)

func translateName(name string, md *descriptor.DescriptorProto) (string, *descriptor.DescriptorProto) {
	// TODO - should really gate this with a test that the marshaller has used json names
	if md != nil {
		for _, f := range md.Field {
			if f.JsonName != nil && f.Name != nil && *f.JsonName == name {
				var subType *descriptor.DescriptorProto

				// If the field has a TypeName then we retrieve the nested type for translating the embedded message names.
				if f.TypeName != nil {
					typeSplit := strings.Split(*f.TypeName, ".")
					typeName := typeSplit[len(typeSplit)-1]
					for _, t := range md.NestedType {
						if typeName == *t.Name {
							subType = t
						}
					}
				}
				return *f.Name, subType
			}
		}
	}
	return name, nil
}

// FieldMaskFromRequestBody creates a FieldMask printing all complete paths from the JSON body.
func FieldMaskFromRequestBody(r io.Reader, md *descriptor.DescriptorProto) (*field_mask.FieldMask, error) {
	fm := &field_mask.FieldMask{}
	var root interface{}
	if err := json.NewDecoder(r).Decode(&root); err != nil {
		if err == io.EOF {
			return fm, nil
		}
		return nil, err
	}

	queue := []fieldMaskPathItem{{node: root, md: md}}
	for len(queue) > 0 {
		// dequeue an item
		item := queue[0]
		queue = queue[1:]

		if m, ok := item.node.(map[string]interface{}); ok {
			// if the item is an object, then enqueue all of its children
			for k, v := range m {
				protoName, subMd := translateName(k, item.md)
				if subMsg, ok := v.(descriptor2.Message); ok {
					_, subMd = descriptor2.ForMessage(subMsg)
				}

				var path string
				if item.path == "" {
					path = protoName
				} else {
					path = item.path + "." + protoName
				}
				queue = append(queue, fieldMaskPathItem{path: path, node: v, md: subMd})
			}
		} else if len(item.path) > 0 {
			// otherwise, it's a leaf node so print its path
			fm.Paths = append(fm.Paths, item.path)
		}
	}

	return fm, nil
}

// fieldMaskPathItem stores a in-progress deconstruction of a path for a fieldmask
type fieldMaskPathItem struct {
	// the list of prior fields leading up to node connected by dots
	path string

	// a generic decoded json object the current item to inspect for further path extraction
	node interface{}

	// descriptor for parent message
	md *descriptor.DescriptorProto
}
