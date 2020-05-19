package runtime

import (
	"encoding/json"
	"io"
	"strings"

	"github.com/golang/protobuf/protoc-gen-go/generator"
	"google.golang.org/genproto/protobuf/field_mask"
)

// FieldMaskFromRequestBody creates a FieldMask printing all complete paths from the JSON body.
func FieldMaskFromRequestBody(r io.Reader) (*field_mask.FieldMask, error) {
	fm := &field_mask.FieldMask{}
	var root interface{}
	if err := json.NewDecoder(r).Decode(&root); err != nil {
		if err == io.EOF {
			return fm, nil
		}
		return nil, err
	}

	queue := []fieldMaskPathItem{{node: root}}
	for len(queue) > 0 {
		// dequeue an item
		item := queue[0]
		queue = queue[1:]

		if m, ok := item.node.(map[string]interface{}); ok {
			// if the item is an object, then enqueue all of its children
			for k, v := range m {
				queue = append(queue, fieldMaskPathItem{path: append(item.path, generator.CamelCase(k)), node: v})
			}
		} else if len(item.path) > 0 {
			// otherwise, it's a leaf node so print its path
			fm.Paths = append(fm.Paths, strings.Join(item.path, "."))
		}
	}

	return fm, nil
}

// fieldMaskPathItem stores a in-progress deconstruction of a path for a fieldmask
type fieldMaskPathItem struct {
	// the list of prior fields leading up to node
	path []string

	// a generic decoded json object the current item to inspect for further path extraction
	node interface{}
}

// CamelCaseFieldMask updates the given FieldMask by converting all of its paths to CamelCase, using the same heuristic
// that's used for naming protobuf fields in Go.
func CamelCaseFieldMask(mask *field_mask.FieldMask) {
	if mask == nil || mask.Paths == nil {
		return
	}

	var newPaths []string
	for _, path := range mask.Paths {
		lowerCasedParts := strings.Split(path, ".")
		var camelCasedParts []string
		for _, part := range lowerCasedParts {
			camelCasedParts = append(camelCasedParts, generator.CamelCase(part))
		}
		newPaths = append(newPaths, strings.Join(camelCasedParts, "."))
	}

	mask.Paths = newPaths
}
