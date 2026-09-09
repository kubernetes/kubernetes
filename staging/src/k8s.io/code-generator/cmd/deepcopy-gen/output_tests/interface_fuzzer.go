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

package outputtests

import (
	"sigs.k8s.io/randfill"

	"k8s.io/code-generator/cmd/deepcopy-gen/output_tests/aliases"
	"k8s.io/code-generator/cmd/deepcopy-gen/output_tests/interfaces"
)

// interfaceFuzzers contains fuzzer that set all interface to nil because our
// JSON deepcopy does not work with it.
// TODO: test also interface deepcopy
var interfaceFuzzers = []interface{}{
	func(s *aliases.AliasAliasInterface, c randfill.Continue) {
		if c.Bool() {
			*s = nil
		} else {
			*s = &aliasAliasInterfaceInstance{X: c.Int()}
		}
	},
	func(s *aliases.AliasInterface, c randfill.Continue) {
		if c.Bool() {
			*s = nil
		} else {
			*s = &aliasAliasInterfaceInstance{X: c.Int()}
		}
	},
	func(s *aliases.Interface, c randfill.Continue) {
		if c.Bool() {
			*s = nil
		} else {
			*s = &aliasAliasInterfaceInstance{X: c.Int()}
		}
	},
	func(s *aliases.AliasInterfaceMap, c randfill.Continue) {
		if c.Bool() {
			*s = nil
		} else {
			*s = make(aliases.AliasInterfaceMap)
			for i := 0; i < c.Intn(3); i++ {
				if c.Bool() {
					(*s)[c.String(0)] = nil
				} else {
					(*s)[c.String(0)] = &aliasAliasInterfaceInstance{X: c.Int()}
				}
			}
		}

	},
	func(s *aliases.AliasInterfaceSlice, c randfill.Continue) {
		if c.Bool() {
			*s = nil
		} else {
			*s = make(aliases.AliasInterfaceSlice, 0)
			for i := 0; i < c.Intn(3); i++ {
				if c.Bool() {
					*s = append(*s, nil)
				} else {
					*s = append(*s, &aliasAliasInterfaceInstance{X: c.Int()})
				}
			}
		}
	},
	func(s *interfaces.Inner, c randfill.Continue) {
		if c.Bool() {
			*s = nil
		} else {
			*s = &interfacesInnerInstance{X: c.Float64()}
		}
	},
}

type aliasAliasInterfaceInstance struct {
	X int
}

func (i *aliasAliasInterfaceInstance) DeepCopyInterface() aliases.Interface {
	if i == nil {
		return nil
	}

	return &aliasAliasInterfaceInstance{X: i.X}
}

func (i *aliasAliasInterfaceInstance) DeepCopyAliasInterface() aliases.AliasInterface {
	if i == nil {
		return nil
	}

	return &aliasAliasInterfaceInstance{X: i.X}
}

func (i *aliasAliasInterfaceInstance) DeepCopyAliasAliasInterface() aliases.AliasAliasInterface {
	if i == nil {
		return nil
	}

	return &aliasAliasInterfaceInstance{X: i.X}
}

type interfacesInnerInstance struct {
	X float64
}

func (i *interfacesInnerInstance) DeepCopyInner() interfaces.Inner {
	if i == nil {
		return nil
	}

	return &interfacesInnerInstance{X: i.X}
}

func (i *interfacesInnerInstance) Function() float64 {
	return i.X
}
