package varnamelen

import "strings"

// stringsValue is the value of a list-of-strings flag.
type stringsValue struct {
	Values []string
}

// declarationsValue is the value of a list-of-declarations flag.
type declarationsValue struct {
	Values []declaration
}

// Set implements Value.
func (sv *stringsValue) Set(values string) error {
	if strings.TrimSpace(values) == "" {
		sv.Values = nil
		return nil
	}

	parts := strings.Split(values, ",")

	sv.Values = make([]string, len(parts))

	for i, part := range parts {
		sv.Values[i] = strings.TrimSpace(part)
	}

	return nil
}

// String implements Value.
func (sv *stringsValue) String() string {
	return strings.Join(sv.Values, ",")
}

// contains returns true if sv contains s.
func (sv *stringsValue) contains(s string) bool {
	for _, v := range sv.Values {
		if v == s {
			return true
		}
	}

	return false
}

// Set implements Value.
func (dv *declarationsValue) Set(values string) error {
	if strings.TrimSpace(values) == "" {
		dv.Values = nil
		return nil
	}

	parts := strings.Split(values, ",")

	dv.Values = make([]declaration, len(parts))

	for idx, part := range parts {
		dv.Values[idx] = parseDeclaration(strings.TrimSpace(part))
	}

	return nil
}

// String implements Value.
func (dv *declarationsValue) String() string {
	parts := make([]string, len(dv.Values))

	for idx, val := range dv.Values {
		parts[idx] = val.name + " " + val.typ
	}

	return strings.Join(parts, ",")
}

// matchVariable returns true if vari matches any of the declarations in dv.
func (dv *declarationsValue) matchVariable(vari variable) bool {
	for _, decl := range dv.Values {
		if vari.match(decl) {
			return true
		}
	}

	return false
}

// matchParameter returns true if param matches any of the declarations in dv.
func (dv *declarationsValue) matchParameter(param parameter) bool {
	for _, decl := range dv.Values {
		if param.match(decl) {
			return true
		}
	}

	return false
}
