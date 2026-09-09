/*
Copyright 2018 The Kubernetes Authors.

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

package admission

type Decorator interface {
	Decorate(handler Interface, name string) Interface
}

type DecoratorFunc func(handler Interface, name string) Interface

func (d DecoratorFunc) Decorate(handler Interface, name string) Interface {
	return d(handler, name)
}

type Decorators []Decorator

// Decorate applies the decorator in inside-out order, i.e. the first decorator in the slice is first applied to the given handler.
func (d Decorators) Decorate(handler Interface, name string) Interface {
	result := handler
	for _, d := range d {
		result = d.Decorate(result, name)
	}

	return result
}
