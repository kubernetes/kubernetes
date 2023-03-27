/*
Copyright 2023 The Kubernetes Authors.

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

package ktesting

import (
	"fmt"

	"github.com/go-logr/logr"
	"sigs.k8s.io/yaml"
)

func Format[T any](obj T) formatAny[T] {
	return formatAny[T]{Object: obj}
}

type formatAny[T any] struct {
	Object T
}

func (f formatAny[T]) String() string {
	yamlBytes, err := yaml.Marshal(&f.Object)
	if err != nil {
		return fmt.Sprintf("error marshaling %T to YAML: %v", f, err)
	}
	return string(yamlBytes)
}

func (f formatAny[T]) MarshalLog() interface{} {
	// Strip implementation of String.
	type noStringer *T
	return noStringer(&f.Object)
}

var _ fmt.Stringer = formatAny[int]{}
var _ logr.Marshaler = formatAny[int]{}
