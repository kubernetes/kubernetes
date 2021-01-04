/*
Copyright 2021 The Kubernetes Authors.

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

package generators

import (
	"strings"

	codegennamer "k8s.io/code-generator/pkg/namer"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
)

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	pluralExceptions := map[string]string{
		"Endpoints": "Endpoints",
	}

	publicPluralNamer := namer.NewPublicPluralNamer(pluralExceptions)

	publicNamer := &ExceptionNamer{
		Exceptions: map[string]string{},
		KeyFunc: func(t *types.Type) string {
			return t.Name.Package + "." + t.Name.Name
		},
		Delegate: namer.NewPublicNamer(0),
	}

	return namer.NameSystems{
		"public":             namer.NewPublicNamer(0),
		"private":            namer.NewPrivateNamer(0),
		"raw":                namer.NewRawNamer("", nil),
		"publicPlural":       publicPluralNamer,
		"allLowercasePlural": namer.NewAllLowercasePluralNamer(pluralExceptions),
		"lowercaseSingular":  &lowercaseSingularNamer{},
		"apiGroup":           codegennamer.NewTagOverrideNamer("publicPlural", publicPluralNamer),
		"versionedClientset": &versionedClientsetNamer{public: publicNamer},
	}
}

// lowercaseSingularNamer implements Namer
type lowercaseSingularNamer struct{}

// Name returns t's name in all lowercase.
func (n *lowercaseSingularNamer) Name(t *types.Type) string {
	return strings.ToLower(t.Name.Name)
}

type versionedClientsetNamer struct {
	public *ExceptionNamer
}

func (r *versionedClientsetNamer) Name(t *types.Type) string {
	// Turns type into a GroupVersion type string based on package.
	parts := strings.Split(t.Name.Package, "/")
	group := parts[len(parts)-2]
	version := parts[len(parts)-1]

	g := r.public.Name(&types.Type{Name: types.Name{Name: group, Package: t.Name.Package}})
	v := r.public.Name(&types.Type{Name: types.Name{Name: version, Package: t.Name.Package}})

	return g + v
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "public"
}

// ExceptionNamer allows you specify exceptional cases with exact names.  This allows you to have control
// for handling various conflicts, like group and resource names for instance.
type ExceptionNamer struct {
	Exceptions map[string]string
	KeyFunc    func(*types.Type) string

	Delegate namer.Namer
}

// Name provides the requested name for a type.
func (n *ExceptionNamer) Name(t *types.Type) string {
	key := n.KeyFunc(t)
	if exception, ok := n.Exceptions[key]; ok {
		return exception
	}
	return n.Delegate.Name(t)
}
