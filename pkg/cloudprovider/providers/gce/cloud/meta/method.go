/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package meta

import (
	"fmt"
	"reflect"
	"strings"
)

func newArg(t reflect.Type) *arg {
	ret := &arg{}

	// Dereference the pointer types to get at the underlying concrete type.
Loop:
	for {
		switch t.Kind() {
		case reflect.Ptr:
			ret.numPtr++
			t = t.Elem()
		default:
			ret.pkg = t.PkgPath()
			ret.typeName += t.Name()
			break Loop
		}
	}
	return ret
}

type arg struct {
	pkg, typeName string
	numPtr        int
}

func (a *arg) normalizedPkg() string {
	if a.pkg == "" {
		return ""
	}

	// Strip the repo.../vendor/ prefix from the package path if present.
	parts := strings.Split(a.pkg, "/")
	// Remove vendor prefix.
	for i := 0; i < len(parts); i++ {
		if parts[i] == "vendor" {
			parts = parts[i+1:]
			break
		}
	}
	switch strings.Join(parts, "/") {
	case "google.golang.org/api/compute/v1":
		return "ga."
	case "google.golang.org/api/compute/v0.alpha":
		return "alpha."
	case "google.golang.org/api/compute/v0.beta":
		return "beta."
	default:
		panic(fmt.Errorf("unhandled package %q", a.pkg))
	}
}

func (a *arg) String() string {
	var ret string
	for i := 0; i < a.numPtr; i++ {
		ret += "*"
	}
	ret += a.normalizedPkg()
	ret += a.typeName
	return ret
}

// newMethod returns a newly initialized method.
func newMethod(s *ServiceInfo, m reflect.Method) *Method {
	ret := &Method{
		ServiceInfo: s,
		m:           m,
		kind:        MethodOperation,
		ReturnType:  "",
	}
	ret.init()
	return ret
}

// MethodKind is the type of method that we are generated code for.
type MethodKind int

const (
	// MethodOperation is a long running method that returns an operation.
	MethodOperation MethodKind = iota
	// MethodGet is a method that immediately returns some data.
	MethodGet MethodKind = iota
	// MethodPaged is a method that returns a paged set of data.
	MethodPaged MethodKind = iota
)

// Method is used to generate the calling code for non-standard methods.
type Method struct {
	*ServiceInfo
	m reflect.Method

	kind MethodKind
	// ReturnType is the return type for the method.
	ReturnType string
	// ItemType is the type of the individual elements returns from a
	// Pages() call. This is only applicable for MethodPaged kind.
	ItemType string
}

// IsOperation is true if the method is an Operation.
func (m *Method) IsOperation() bool {
	return m.kind == MethodOperation
}

// IsPaged is true if the method paged.
func (m *Method) IsPaged() bool {
	return m.kind == MethodPaged
}

// IsGet is true if the method simple get.
func (m *Method) IsGet() bool {
	return m.kind == MethodGet
}

// argsSkip is the number of arguments to skip when generating the
// synthesized method.
func (m *Method) argsSkip() int {
	switch m.keyType {
	case Zonal:
		return 4
	case Regional:
		return 4
	case Global:
		return 3
	}
	panic(fmt.Errorf("invalid KeyType %v", m.keyType))
}

// args return a list of arguments to the method, skipping the first skip
// elements. If nameArgs is true, then the arguments will include a generated
// parameter name (arg<N>). prefix will be added to the parameters.
func (m *Method) args(skip int, nameArgs bool, prefix []string) []string {
	var args []*arg
	fType := m.m.Func.Type()
	for i := 0; i < fType.NumIn(); i++ {
		t := fType.In(i)
		args = append(args, newArg(t))
	}

	var a []string
	for i := skip; i < fType.NumIn(); i++ {
		if nameArgs {
			a = append(a, fmt.Sprintf("arg%d %s", i-skip, args[i]))
		} else {
			a = append(a, args[i].String())
		}
	}
	return append(prefix, a...)
}

// init the method. This performs some rudimentary static checking as well as
// determines the kind of method by looking at the shape (method signature) of
// the object.
func (m *Method) init() {
	fType := m.m.Func.Type()
	if fType.NumIn() < m.argsSkip() {
		err := fmt.Errorf("method %q.%q, arity = %d which is less than required (< %d)",
			m.Service, m.Name(), fType.NumIn(), m.argsSkip())
		panic(err)
	}
	// Skipped args should all be string (they will be projectID, zone, region etc).
	for i := 1; i < m.argsSkip(); i++ {
		if fType.In(i).Kind() != reflect.String {
			panic(fmt.Errorf("method %q.%q: skipped args can only be strings", m.Service, m.Name()))
		}
	}
	// Return of the method must return a single value of type *xxxCall.
	if fType.NumOut() != 1 || fType.Out(0).Kind() != reflect.Ptr || !strings.HasSuffix(fType.Out(0).Elem().Name(), "Call") {
		panic(fmt.Errorf("method %q.%q: generator only supports methods returning an *xxxCall object",
			m.Service, m.Name()))
	}
	returnType := fType.Out(0)
	returnTypeName := fType.Out(0).Elem().Name()
	// xxxCall must have a Do() method.
	doMethod, ok := returnType.MethodByName("Do")
	if !ok {
		panic(fmt.Errorf("method %q.%q: return type %q does not have a Do() method",
			m.Service, m.Name(), returnTypeName))
	}
	_, hasPages := returnType.MethodByName("Pages")
	// Do() method must return (*T, error).
	switch doMethod.Func.Type().NumOut() {
	case 2:
		out0 := doMethod.Func.Type().Out(0)
		if out0.Kind() != reflect.Ptr {
			panic(fmt.Errorf("method %q.%q: return type %q of Do() = S, _; S must be pointer type (%v)",
				m.Service, m.Name(), returnTypeName, out0))
		}
		m.ReturnType = out0.Elem().Name()
		switch {
		case out0.Elem().Name() == "Operation":
			m.kind = MethodOperation
		case hasPages:
			m.kind = MethodPaged
			// Pages() returns a xxxList that has the actual list
			// of objects in the xxxList.Items field.
			listType := out0.Elem()
			itemsField, ok := listType.FieldByName("Items")
			if !ok {
				panic(fmt.Errorf("method %q.%q: paged return type %q does not have a .Items field", m.Service, m.Name(), listType.Name()))
			}
			// itemsField will be a []*ItemType. Dereference to
			// extract the ItemType.
			itemsType := itemsField.Type
			if itemsType.Kind() != reflect.Slice && itemsType.Elem().Kind() != reflect.Ptr {
				panic(fmt.Errorf("method %q.%q: paged return type %q.Items is not an array of pointers", m.Service, m.Name(), listType.Name()))
			}
			m.ItemType = itemsType.Elem().Elem().Name()
		default:
			m.kind = MethodGet
		}
		// Second argument must be "error".
		if doMethod.Func.Type().Out(1).Name() != "error" {
			panic(fmt.Errorf("method %q.%q: return type %q of Do() = S, T; T must be 'error'",
				m.Service, m.Name(), returnTypeName))
		}
		break
	default:
		panic(fmt.Errorf("method %q.%q: %q Do() return type is not handled by the generator",
			m.Service, m.Name(), returnTypeName))
	}
}

// Name is the name of the method.
func (m *Method) Name() string {
	return m.m.Name
}

// CallArgs is a list of comma separated "argN" used for calling the method.
// For example, if the method has two additional arguments, this will return
// "arg0, arg1".
func (m *Method) CallArgs() string {
	var args []string
	for i := m.argsSkip(); i < m.m.Func.Type().NumIn(); i++ {
		args = append(args, fmt.Sprintf("arg%d", i-m.argsSkip()))
	}
	if len(args) == 0 {
		return ""
	}
	return fmt.Sprintf(", %s", strings.Join(args, ", "))
}

// MockHookName is the name of the hook function in the mock.
func (m *Method) MockHookName() string {
	return m.m.Name + "Hook"
}

// MockHook is the definition of the hook function.
func (m *Method) MockHook() string {
	args := m.args(m.argsSkip(), false, []string{
		"context.Context",
		"*meta.Key",
	})
	if m.kind == MethodPaged {
		args = append(args, "*filter.F")
	}

	args = append(args, fmt.Sprintf("*%s", m.MockWrapType()))

	switch m.kind {
	case MethodOperation:
		return fmt.Sprintf("%v func(%v) error", m.MockHookName(), strings.Join(args, ", "))
	case MethodGet:
		return fmt.Sprintf("%v func(%v) (*%v.%v, error)", m.MockHookName(), strings.Join(args, ", "), m.Version(), m.ReturnType)
	case MethodPaged:
		return fmt.Sprintf("%v func(%v) ([]*%v.%v, error)", m.MockHookName(), strings.Join(args, ", "), m.Version(), m.ItemType)
	default:
		panic(fmt.Errorf("invalid method kind: %v", m.kind))
	}
}

// FcnArgs is the function signature for the definition of the method.
func (m *Method) FcnArgs() string {
	args := m.args(m.argsSkip(), true, []string{
		"ctx context.Context",
		"key *meta.Key",
	})
	if m.kind == MethodPaged {
		args = append(args, "fl *filter.F")
	}

	switch m.kind {
	case MethodOperation:
		return fmt.Sprintf("%v(%v) error", m.m.Name, strings.Join(args, ", "))
	case MethodGet:
		return fmt.Sprintf("%v(%v) (*%v.%v, error)", m.m.Name, strings.Join(args, ", "), m.Version(), m.ReturnType)
	case MethodPaged:
		return fmt.Sprintf("%v(%v) ([]*%v.%v, error)", m.m.Name, strings.Join(args, ", "), m.Version(), m.ItemType)
	default:
		panic(fmt.Errorf("invalid method kind: %v", m.kind))
	}
}

// InterfaceFunc is the function declaration of the method in the interface.
func (m *Method) InterfaceFunc() string {
	args := []string{
		"context.Context",
		"*meta.Key",
	}
	args = m.args(m.argsSkip(), false, args)
	if m.kind == MethodPaged {
		args = append(args, "*filter.F")
	}

	switch m.kind {
	case MethodOperation:
		return fmt.Sprintf("%v(%v) error", m.m.Name, strings.Join(args, ", "))
	case MethodGet:
		return fmt.Sprintf("%v(%v) (*%v.%v, error)", m.m.Name, strings.Join(args, ", "), m.Version(), m.ReturnType)
	case MethodPaged:
		return fmt.Sprintf("%v(%v) ([]*%v.%v, error)", m.m.Name, strings.Join(args, ", "), m.Version(), m.ItemType)
	default:
		panic(fmt.Errorf("invalid method kind: %v", m.kind))
	}
}
