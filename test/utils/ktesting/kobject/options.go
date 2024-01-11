/*
Copyright 2024 The Kubernetes Authors.

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

package kobject

import (
	"k8s.io/kubernetes/test/utils/ktesting"
)

// WithNamespace sets a default namespace name for operations using the context.
func WithNamespace(tCtx ktesting.TContext, namespace string) ktesting.TContext {
	opt := getOptions(tCtx)
	opt.namespace = namespace
	return setOptions(tCtx, opt)
}

// Namespace returns the default namespace stored in the context.
// May be empty.
func Namespace(tCtx ktesting.TContext) string {
	opt := getOptions(tCtx)
	return opt.namespace
}

func getOptions(tCtx ktesting.TContext) options {
	value := tCtx.Value(optionsKey)
	if opt, ok := value.(options); ok {
		return opt
	}
	return options{}
}

func setOptions(tCtx ktesting.TContext, opt options) ktesting.TContext {
	return ktesting.WithValue(tCtx, optionsKey, opt)
}

type optionsKeyType struct{}

var optionsKey optionsKeyType

type options struct {
	namespace string
}
