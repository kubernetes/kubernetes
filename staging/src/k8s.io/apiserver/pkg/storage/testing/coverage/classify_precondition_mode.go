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

package coverage

import (
	"k8s.io/apiserver/pkg/storage"
)

// PreconditionMode classifies a Delete call's *storage.Preconditions.
type PreconditionMode string

const (
	PreconditionNone            PreconditionMode = "PreconditionNone"
	PreconditionUID             PreconditionMode = "PreconditionUID"
	PreconditionResourceVersion PreconditionMode = "PreconditionResourceVersion"
	PreconditionBoth            PreconditionMode = "PreconditionBoth"
)

func classifyPreconditionMode(p *storage.Preconditions) PreconditionMode {
	switch {
	case p == nil:
		return PreconditionNone
	case p.UID != nil && p.ResourceVersion != nil:
		return PreconditionBoth
	case p.UID != nil:
		return PreconditionUID
	case p.ResourceVersion != nil:
		return PreconditionResourceVersion
	default:
		return PreconditionNone
	}
}
