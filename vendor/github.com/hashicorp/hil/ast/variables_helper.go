package ast

import "fmt"

func VariableListElementTypesAreHomogenous(variableName string, list []Variable) (Type, error) {
	listTypes := make(map[Type]struct{})
	for _, v := range list {
		if _, ok := listTypes[v.Type]; ok {
			continue
		}
		listTypes[v.Type] = struct{}{}
	}

	if len(listTypes) != 1 && len(list) != 0 {
		return TypeInvalid, fmt.Errorf("list %q does not have homogenous types. found %s", variableName, reportTypes(listTypes))
	}

	if len(list) > 0 {
		return list[0].Type, nil
	}

	return TypeInvalid, fmt.Errorf("list %q does not have any elements so cannot determine type.", variableName)
}

func VariableMapValueTypesAreHomogenous(variableName string, vmap map[string]Variable) (Type, error) {
	valueTypes := make(map[Type]struct{})
	for _, v := range vmap {
		if _, ok := valueTypes[v.Type]; ok {
			continue
		}
		valueTypes[v.Type] = struct{}{}
	}

	if len(valueTypes) != 1 && len(vmap) != 0 {
		return TypeInvalid, fmt.Errorf("map %q does not have homogenous value types. found %s", variableName, reportTypes(valueTypes))
	}

	// For loop here is an easy way to get a single key, we return immediately.
	for _, v := range vmap {
		return v.Type, nil
	}

	// This means the map is empty
	return TypeInvalid, fmt.Errorf("map %q does not have any elements so cannot determine type.", variableName)
}
