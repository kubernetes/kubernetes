package ini

import (
	"fmt"
	"sort"
)

// Visitor is an interface used by walkers that will
// traverse an array of ASTs.
type Visitor interface {
	VisitExpr(AST) error
	VisitStatement(AST) error
}

// DefaultVisitor is used to visit statements and expressions
// and ensure that they are both of the correct format.
// In addition, upon visiting this will build sections and populate
// the Sections field which can be used to retrieve profile
// configuration.
type DefaultVisitor struct {
	scope    string
	Sections Sections
}

// NewDefaultVisitor return a DefaultVisitor
func NewDefaultVisitor() *DefaultVisitor {
	return &DefaultVisitor{
		Sections: Sections{
			container: map[string]Section{},
		},
	}
}

// VisitExpr visits expressions...
func (v *DefaultVisitor) VisitExpr(expr AST) error {
	t := v.Sections.container[v.scope]
	if t.values == nil {
		t.values = values{}
	}

	switch expr.Kind {
	case ASTKindExprStatement:
		opExpr := expr.GetRoot()
		switch opExpr.Kind {
		case ASTKindEqualExpr:
			children := opExpr.GetChildren()
			if len(children) <= 1 {
				return NewParseError("unexpected token type")
			}

			rhs := children[1]

			// The right-hand value side the equality expression is allowed to contain '[', ']', ':', '=' in the values.
			// If the token is not either a literal or one of the token types that identifies those four additional
			// tokens then error.
			if !(rhs.Root.Type() == TokenLit || rhs.Root.Type() == TokenOp || rhs.Root.Type() == TokenSep) {
				return NewParseError("unexpected token type")
			}

			key := EqualExprKey(opExpr)
			v, err := newValue(rhs.Root.ValueType, rhs.Root.base, rhs.Root.Raw())
			if err != nil {
				return err
			}

			t.values[key] = v
		default:
			return NewParseError(fmt.Sprintf("unsupported expression %v", expr))
		}
	default:
		return NewParseError(fmt.Sprintf("unsupported expression %v", expr))
	}

	v.Sections.container[v.scope] = t
	return nil
}

// VisitStatement visits statements...
func (v *DefaultVisitor) VisitStatement(stmt AST) error {
	switch stmt.Kind {
	case ASTKindCompletedSectionStatement:
		child := stmt.GetRoot()
		if child.Kind != ASTKindSectionStatement {
			return NewParseError(fmt.Sprintf("unsupported child statement: %T", child))
		}

		name := string(child.Root.Raw())
		v.Sections.container[name] = Section{}
		v.scope = name
	default:
		return NewParseError(fmt.Sprintf("unsupported statement: %s", stmt.Kind))
	}

	return nil
}

// Sections is a map of Section structures that represent
// a configuration.
type Sections struct {
	container map[string]Section
}

// GetSection will return section p. If section p does not exist,
// false will be returned in the second parameter.
func (t Sections) GetSection(p string) (Section, bool) {
	v, ok := t.container[p]
	return v, ok
}

// values represents a map of union values.
type values map[string]Value

// List will return a list of all sections that were successfully
// parsed.
func (t Sections) List() []string {
	keys := make([]string, len(t.container))
	i := 0
	for k := range t.container {
		keys[i] = k
		i++
	}

	sort.Strings(keys)
	return keys
}

// Section contains a name and values. This represent
// a sectioned entry in a configuration file.
type Section struct {
	Name   string
	values values
}

// Has will return whether or not an entry exists in a given section
func (t Section) Has(k string) bool {
	_, ok := t.values[k]
	return ok
}

// ValueType will returned what type the union is set to. If
// k was not found, the NoneType will be returned.
func (t Section) ValueType(k string) (ValueType, bool) {
	v, ok := t.values[k]
	return v.Type, ok
}

// Bool returns a bool value at k
func (t Section) Bool(k string) bool {
	return t.values[k].BoolValue()
}

// Int returns an integer value at k
func (t Section) Int(k string) int64 {
	return t.values[k].IntValue()
}

// Float64 returns a float value at k
func (t Section) Float64(k string) float64 {
	return t.values[k].FloatValue()
}

// String returns the string value at k
func (t Section) String(k string) string {
	_, ok := t.values[k]
	if !ok {
		return ""
	}
	return t.values[k].StringValue()
}
