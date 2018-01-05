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

	"github.com/golang/glog"
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
	ret := &Method{s, m, ""}
	ret.init()
	return ret
}

// Method is used to generate the calling code non-standard methods.
type Method struct {
	*ServiceInfo
	m reflect.Method

	ReturnType string
}

// argsSkip is the number of arguments to skip when generating the
// synthesized method.
func (mr *Method) argsSkip() int {
	switch mr.keyType {
	case Zonal:
		return 4
	case Regional:
		return 4
	case Global:
		return 3
	}
	panic(fmt.Errorf("invalid KeyType %v", mr.keyType))
}

// args return a list of arguments to the method, skipping the first skip
// elements. If nameArgs is true, then the arguments will include a generated
// parameter name (arg<N>). prefix will be added to the parameters.
func (mr *Method) args(skip int, nameArgs bool, prefix []string) []string {
	var args []*arg
	fType := mr.m.Func.Type()
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

func (mr *Method) init() {
	fType := mr.m.Func.Type()
	if fType.NumIn() < mr.argsSkip() {
		err := fmt.Errorf("method %q.%q, arity = %d which is less than required (< %d)",
			mr.Service, mr.Name(), fType.NumIn(), mr.argsSkip())
		panic(err)
	}
	// Skipped args should all be string (they will be projectID, zone, region etc).
	for i := 1; i < mr.argsSkip(); i++ {
		if fType.In(i).Kind() != reflect.String {
			panic(fmt.Errorf("method %q.%q: skipped args can only be strings", mr.Service, mr.Name()))
		}
	}
	// Return of the method must return a single value of type *xxxCall.
	if fType.NumOut() != 1 || fType.Out(0).Kind() != reflect.Ptr || !strings.HasSuffix(fType.Out(0).Elem().Name(), "Call") {
		panic(fmt.Errorf("method %q.%q: generator only supports methods returning an *xxxCall object",
			mr.Service, mr.Name()))
	}
	returnType := fType.Out(0)
	returnTypeName := fType.Out(0).Elem().Name()
	// xxxCall must have a Do() method.
	doMethod, ok := returnType.MethodByName("Do")
	if !ok {
		panic(fmt.Errorf("method %q.%q: return type %q does not have a Do() method",
			mr.Service, mr.Name(), returnTypeName))
	}
	// Do() method must return (*T, error).
	switch doMethod.Func.Type().NumOut() {
	case 2:
		glog.Infof("Method %q.%q: return type %q of Do() = %v, %v",
			mr.Service, mr.Name(), returnTypeName, doMethod.Func.Type().Out(0), doMethod.Func.Type().Out(1))
		out0 := doMethod.Func.Type().Out(0)
		if out0.Kind() != reflect.Ptr {
			panic(fmt.Errorf("method %q.%q: return type %q of Do() = S, _; S must be pointer type (%v)",
				mr.Service, mr.Name(), returnTypeName, out0))
		}
		mr.ReturnType = out0.Elem().Name()
		if out0.Elem().Name() == "Operation" {
			glog.Infof("Method %q.%q is an *Operation", mr.Service, mr.Name())
		} else {
			glog.Infof("Method %q.%q returns %v", mr.Service, mr.Name(), out0)
		}
		// Second argument must be "error".
		if doMethod.Func.Type().Out(1).Name() != "error" {
			panic(fmt.Errorf("method %q.%q: return type %q of Do() = S, T; T must be 'error'",
				mr.Service, mr.Name(), returnTypeName))
		}
		break
	default:
		panic(fmt.Errorf("method %q.%q: %q Do() return type is not handled by the generator",
			mr.Service, mr.Name(), returnTypeName))
	}
}

func (mr *Method) Name() string {
	return mr.m.Name
}

func (mr *Method) CallArgs() string {
	var args []string
	for i := mr.argsSkip(); i < mr.m.Func.Type().NumIn(); i++ {
		args = append(args, fmt.Sprintf("arg%d", i-mr.argsSkip()))
	}
	if len(args) == 0 {
		return ""
	}
	return fmt.Sprintf(", %s", strings.Join(args, ", "))
}

func (mr *Method) MockHookName() string {
	return mr.m.Name + "Hook"
}

func (mr *Method) MockHook() string {
	args := mr.args(mr.argsSkip(), false, []string{
		fmt.Sprintf("*%s", mr.MockWrapType()),
		"context.Context",
		"meta.Key",
	})
	if mr.ReturnType == "Operation" {
		return fmt.Sprintf("%v func(%v) error", mr.MockHookName(), strings.Join(args, ", "))
	}
	return fmt.Sprintf("%v func(%v) (*%v.%v, error)", mr.MockHookName(), strings.Join(args, ", "), mr.Version(), mr.ReturnType)
}

func (mr *Method) FcnArgs() string {
	args := mr.args(mr.argsSkip(), true, []string{
		"ctx context.Context",
		"key meta.Key",
	})

	if mr.ReturnType == "Operation" {
		return fmt.Sprintf("%v(%v) error", mr.m.Name, strings.Join(args, ", "))
	}
	return fmt.Sprintf("%v(%v) (*%v.%v, error)", mr.m.Name, strings.Join(args, ", "), mr.Version(), mr.ReturnType)
}

func (mr *Method) InterfaceFunc() string {
	args := mr.args(mr.argsSkip(), false, []string{"context.Context", "meta.Key"})
	if mr.ReturnType == "Operation" {
		return fmt.Sprintf("%v(%v) error", mr.m.Name, strings.Join(args, ", "))
	}
	return fmt.Sprintf("%v(%v) (*%v.%v, error)", mr.m.Name, strings.Join(args, ", "), mr.Version(), mr.ReturnType)
}
