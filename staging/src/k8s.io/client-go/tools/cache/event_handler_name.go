/*
Copyright The Kubernetes Authors.

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

package cache

import (
	"fmt"
	"reflect"
	"runtime"
	"strings"
)

func nameForHandler(handler ResourceEventHandler) (name string) {
	defer func() {
		// Last resort: let Sprintf handle it.
		if name == "" {
			name = fmt.Sprintf("%T", handler)
		}
	}()

	if handler == nil {
		return ""
	}
	switch handler := handler.(type) {
	case *ResourceEventHandlerFuncs:
		return nameForHandlerFuncs(*handler)
	case ResourceEventHandlerFuncs:
		return nameForHandlerFuncs(handler)
	default:
		// We can use the fully qualified name of whatever
		// provides the interface. We don't care whether
		// it contains fields or methods which provide
		// the interface methods.
		value := reflect.ValueOf(handler)
		if value.Type().Kind() == reflect.Interface {
			// Probably not needed, but let's play it safe.
			value = value.Elem()
		}
		if value.Type().Kind() == reflect.Pointer {
			if !value.IsNil() {
				value = value.Elem()
			}
		}
		name := value.Type().PkgPath()
		if name != "" {
			name += "."
		}
		if typeName := value.Type().Name(); typeName != "" {
			name += typeName
		}
		return name
	}
}

func nameForHandlerFuncs(funcs ResourceEventHandlerFuncs) string {
	return nameForFunctions(funcs.AddFunc, funcs.UpdateFunc, funcs.DeleteFunc)
}

func nameForFunctions(fs ...any) string {
	// If all functions are defined in the same place, then we
	// don't care about the actual function name in
	// e.g. "main.FuncName" or "main.(*Foo).FuncName-fm", instead
	// we use the common qualifier.
	//
	// But we don't know that yet, so we also collect all names.
	var qualifier string
	singleQualifier := true
	var names []string
	for _, f := range fs {
		if f == nil {
			continue
		}
		name := nameForFunction(f)
		if name == "" {
			continue
		}
		names = append(names, name)

		newQualifier := name
		index := strings.LastIndexByte(newQualifier, '.')
		if index > 0 {
			newQualifier = newQualifier[:index]
		}
		switch qualifier {
		case "":
			qualifier = newQualifier
		case newQualifier:
			// So far, so good...
		default:
			// Nope, different.
			singleQualifier = false
		}
	}

	if singleQualifier {
		return qualifier
	}

	return strings.Join(names, "+")
}

func nameForFunction(f any) string {
	fn := runtime.FuncForPC(reflect.ValueOf(f).Pointer())
	if fn == nil {
		return ""
	}
	return fn.Name()
}
