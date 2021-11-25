package starlarkstruct

import (
	"fmt"

	"go.starlark.net/starlark"
)

// A Module is a named collection of values,
// typically a suite of functions imported by a load statement.
//
// It differs from Struct primarily in that its string representation
// does not enumerate its fields.
type Module struct {
	Name    string
	Members starlark.StringDict
}

var _ starlark.HasAttrs = (*Module)(nil)

func (m *Module) Attr(name string) (starlark.Value, error) { return m.Members[name], nil }
func (m *Module) AttrNames() []string                      { return m.Members.Keys() }
func (m *Module) Freeze()                                  { m.Members.Freeze() }
func (m *Module) Hash() (uint32, error)                    { return 0, fmt.Errorf("unhashable: %s", m.Type()) }
func (m *Module) String() string                           { return fmt.Sprintf("<module %q>", m.Name) }
func (m *Module) Truth() starlark.Bool                     { return true }
func (m *Module) Type() string                             { return "module" }

// MakeModule may be used as the implementation of a Starlark built-in
// function, module(name, **kwargs). It returns a new module with the
// specified name and members.
func MakeModule(thread *starlark.Thread, b *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (starlark.Value, error) {
	var name string
	if err := starlark.UnpackPositionalArgs(b.Name(), args, nil, 1, &name); err != nil {
		return nil, err
	}
	members := make(starlark.StringDict, len(kwargs))
	for _, kwarg := range kwargs {
		k := string(kwarg[0].(starlark.String))
		members[k] = kwarg[1]
	}
	return &Module{name, members}, nil
}
