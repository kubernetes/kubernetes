package rules

import (
	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/types"
)

const ListTypeIDLTag = "listType"

// ListTypeMissing implements APIRule interface.
// A list type is required for inlined list.
type ListTypeMissing struct{}

// Name returns the name of APIRule
func (l *ListTypeMissing) Name() string {
	return "list_type_missing"
}

// Validate evaluates API rule on type t and returns a list of field names in
// the type that violate the rule. Empty field name [""] implies the entire
// type violates the rule.
func (l *ListTypeMissing) Validate(t *types.Type) ([]string, error) {
	fields := make([]string, 0)

	switch t.Kind {
	case types.Struct:
		for _, m := range t.Members {
			hasListType := gengo.ExtractCommentTags("+", m.CommentLines)[ListTypeIDLTag] != nil

			if m.Name == "Items" && m.Type.Kind == types.Slice && hasNamedMember(t, "ListMeta") {
				if hasListType {
					fields = append(fields, m.Name)
				}
				continue
			}

			// All slice fields must have a list-type tag except []byte
			if m.Type.Kind == types.Slice && m.Type.Elem != types.Byte && !hasListType {
				fields = append(fields, m.Name)
				continue
			}
		}
	}

	return fields, nil
}

func hasNamedMember(t *types.Type, name string) bool {
	for _, m := range t.Members {
		if m.Name == name {
			return true
		}
	}
	return false
}
