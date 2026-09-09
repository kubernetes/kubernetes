package cel

import (
	"slices"
	"strings"

	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/types"
)

// fieldPath represents a selection path to a field from a variable in a CEL environment.
type fieldPath struct {
	celType *Type
	// path represents the selection path to the field.
	path        string
	description string
	isLeaf      bool
}

// Documentation implements the Documentor interface.
func (f *fieldPath) Documentation() *common.Doc {
	return common.NewFieldDoc(f.path, f.celType.String(), f.description)
}

type documentationProvider interface {
	// FindStructFieldDescription returns documentation for a field if available.
	// Returns false if the field could not be found.
	FindStructFieldDescription(typeName, fieldName string) (string, bool)
}

type backtrack struct {
	// provider used to resolve types.
	provider types.Provider
	// paths of fields that have been visited along the path.
	path []string
	// types of fields that have been visited along the path. used to avoid cycles.
	types []*Type
}

func (b *backtrack) push(pathStep string, celType *Type) {
	b.path = append(b.path, pathStep)
	b.types = append(b.types, celType)
}

func (b *backtrack) pop() {
	b.path = b.path[:len(b.path)-1]
	b.types = b.types[:len(b.types)-1]
}

func formatPath(path []string) string {
	var buffer strings.Builder
	for i, p := range path {
		if i == 0 {
			buffer.WriteString(p)
			continue
		}
		if strings.HasPrefix(p, "[") {
			buffer.WriteString(p)
			continue
		}
		buffer.WriteString(".")
		buffer.WriteString(p)
	}
	return buffer.String()
}

func (b *backtrack) expandFieldPaths(celType *Type, paths []*fieldPath) []*fieldPath {
	if slices.ContainsFunc(b.types[:len(b.types)-1], func(t *Type) bool { return t.String() == celType.String() }) {
		// Cycle detected, so stop expanding.
		paths[len(paths)-1].isLeaf = false
		return paths
	}
	switch celType.Kind() {
	case types.StructKind:
		fields, ok := b.provider.FindStructFieldNames(celType.String())
		if !ok {
			// Caller added this type to the path, so it must be a leaf.
			paths[len(paths)-1].isLeaf = true
			return paths
		}
		for _, field := range fields {
			fieldType, ok := b.provider.FindStructFieldType(celType.String(), field)
			if !ok {
				// Field not found, either hidden or an error.
				continue
			}
			b.push(field, celType)
			description := ""
			if docProvider, ok := b.provider.(documentationProvider); ok {
				description, _ = docProvider.FindStructFieldDescription(celType.String(), field)
			}
			path := &fieldPath{
				celType:     fieldType.Type,
				path:        formatPath(b.path),
				description: description,
				isLeaf:      false,
			}
			paths = append(paths, path)
			paths = b.expandFieldPaths(fieldType.Type, paths)
			b.pop()
		}
		return paths
	case types.MapKind:
		if len(celType.Parameters()) != 2 {
			// dynamic map, so treat as a leaf.
			paths[len(paths)-1].isLeaf = true
			return paths
		}
		mapKeyType := celType.Parameters()[0]
		mapValueType := celType.Parameters()[1]
		// Add a placeholder for the map key kind (the zero value).
		keyIdentifier := ""
		switch mapKeyType.Kind() {
		case types.StringKind:
			keyIdentifier = "[\"\"]"
		case types.IntKind:
			keyIdentifier = "[0]"
		case types.UintKind:
			keyIdentifier = "[0u]"
		case types.BoolKind:
			keyIdentifier = "[false]"
		default:
			// Caller added this type to the path, so it must be a leaf.
			paths[len(paths)-1].isLeaf = true
			return paths
		}
		b.push(keyIdentifier, mapValueType)
		defer b.pop()
		return b.expandFieldPaths(mapValueType, paths)
	case types.ListKind:
		if len(celType.Parameters()) != 1 {
			// dynamic list, so treat as a leaf.
			paths[len(paths)-1].isLeaf = true
			return paths
		}
		listElemType := celType.Parameters()[0]
		b.push("[0]", listElemType)
		defer b.pop()
		return b.expandFieldPaths(listElemType, paths)
	default:
		paths[len(paths)-1].isLeaf = true
	}

	return paths
}

// fieldPathsForType expands the reachable fields from the given root identifier.
func fieldPathsForType(provider types.Provider, identifier string, celType *Type) []*fieldPath {
	b := &backtrack{
		provider: provider,
		path:     []string{identifier},
		types:    []*Type{celType},
	}
	paths := []*fieldPath{
		{
			celType: celType,
			path:    identifier,
			isLeaf:  false,
		},
	}

	return b.expandFieldPaths(celType, paths)
}
