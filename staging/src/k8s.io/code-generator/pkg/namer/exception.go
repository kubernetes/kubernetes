/*
Copyright 2019 The Kubernetes Authors.

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

package namer

import (
	gengonamer "k8s.io/gengo/namer"

	"k8s.io/gengo/types"
)

// ExceptionNamer allows you specify exceptional cases with exact names.  This allows you to have control
// for handling various conflicts, like group and resource names for instance.
type ExceptionNamer struct {
	Exceptions map[string]string
	KeyFunc    func(*types.Type) string

	Delegate gengonamer.Namer
}

// Name provides the requested name for a type.
func (n *ExceptionNamer) Name(t *types.Type) string {
	key := n.KeyFunc(t)
	if exception, ok := n.Exceptions[key]; ok {
		return exception
	}
	return n.Delegate.Name(t)
}
