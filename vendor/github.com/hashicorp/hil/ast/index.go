package ast

import (
	"fmt"
	"strings"
)

// Index represents an indexing operation into another data structure
type Index struct {
	Target Node
	Key    Node
	Posx   Pos
}

func (n *Index) Accept(v Visitor) Node {
	return v(n)
}

func (n *Index) Pos() Pos {
	return n.Posx
}

func (n *Index) String() string {
	return fmt.Sprintf("Index(%s, %s)", n.Target, n.Key)
}

func (n *Index) Type(s Scope) (Type, error) {
	variableAccess, ok := n.Target.(*VariableAccess)
	if !ok {
		return TypeInvalid, fmt.Errorf("target is not a variable")
	}

	variable, ok := s.LookupVar(variableAccess.Name)
	if !ok {
		return TypeInvalid, fmt.Errorf("unknown variable accessed: %s", variableAccess.Name)
	}

	switch variable.Type {
	case TypeList:
		return n.typeList(variable, variableAccess.Name)
	case TypeMap:
		return n.typeMap(variable, variableAccess.Name)
	default:
		return TypeInvalid, fmt.Errorf("invalid index operation into non-indexable type: %s", variable.Type)
	}
}

func (n *Index) typeList(variable Variable, variableName string) (Type, error) {
	// We assume type checking has already determined that this is a list
	list := variable.Value.([]Variable)

	return VariableListElementTypesAreHomogenous(variableName, list)
}

func (n *Index) typeMap(variable Variable, variableName string) (Type, error) {
	// We assume type checking has already determined that this is a map
	vmap := variable.Value.(map[string]Variable)

	return VariableMapValueTypesAreHomogenous(variableName, vmap)
}

func reportTypes(typesFound map[Type]struct{}) string {
	stringTypes := make([]string, len(typesFound))
	i := 0
	for k, _ := range typesFound {
		stringTypes[0] = k.String()
		i++
	}
	return strings.Join(stringTypes, ", ")
}

func (n *Index) GoString() string {
	return fmt.Sprintf("*%#v", *n)
}
