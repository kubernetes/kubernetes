/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package util

import (
	"fmt"
	"reflect"

	"github.com/golang/glog"

	v1gen "k8s.io/kubernetes/pkg/api/v1/generated"
)

// This is a map from Type to a function which will apply defaults to an
// instance of that type.
var defaultableTypes = map[reflect.Type]func(obj interface{}){}

func init() {
	v1gen.InitDefaultableTypes(registerDefaultableType)
}

func registerDefaultableType(t reflect.Type, fn func(obj interface{})) {
	//FIXME: demand pointer
	glog.Infof("registered defaultable type %s.%s", t.Elem().PkgPath(), t.Elem().Name())
	if _, found := defaultableTypes[t]; found {
		panic(fmt.Sprintf("repeated call to RegisterDefaultableType(%s.%s)", t.PkgPath(), t.Name()))
	}
	defaultableTypes[t] = fn
}

// Apply defaults to the passed object.  This is done with a (code-generated)
// recursive walk of the type, looking for fields with an ApplyDefaults()
// method.
func ApplyDefaults(obj interface{}) {
	//FIXME: demand pointer
	if fn := defaultableTypes[reflect.TypeOf(obj)]; fn != nil {
		fn(obj)
	}
}
