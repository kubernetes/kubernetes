package rules

import (
	"fmt"

	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/types"
)

const ListTypeIDLTag = "listType"

// K8sListTypeIDLTag is the declarative validation spelling of ListTypeIDLTag.
const K8sListTypeIDLTag = "k8s:listType"

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
			tags, err := gengo.ExtractFunctionStyleCommentTags("+", []string{ListTypeIDLTag, K8sListTypeIDLTag}, m.CommentLines)
			if err != nil {
				return nil, fmt.Errorf("%v.%v: %w", t.Name, m.Name, err)
			}
			hasListType := tags[ListTypeIDLTag] != nil || tags[K8sListTypeIDLTag] != nil

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
