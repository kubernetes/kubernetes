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
	if variable.Type != TypeList {
		return TypeInvalid, fmt.Errorf("invalid index operation into non-indexable type: %s", variable.Type)
	}

	list := variable.Value.([]Variable)

	// Ensure that the types of the list elements are homogenous
	listTypes := make(map[Type]struct{})
	for _, v := range list {
		if _, ok := listTypes[v.Type]; ok {
			continue
		}
		listTypes[v.Type] = struct{}{}
	}

	if len(listTypes) != 1 {
		typesFound := make([]string, len(listTypes))
		i := 0
		for k, _ := range listTypes {
			typesFound[0] = k.String()
			i++
		}
		types := strings.Join(typesFound, ", ")
		return TypeInvalid, fmt.Errorf("list %q does not have homogenous types. found %s", variableAccess.Name, types)
	}

	return list[0].Type, nil
}

func (n *Index) GoString() string {
	return fmt.Sprintf("*%#v", *n)
}
