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

package apidefinitions

import "k8s.io/apimachinery/pkg/util/sets"

// Spec describes one generator's tag protocol. Required for backward compatibility
// with existing tag behavior.
type Spec struct {
	// ActivationTag declare the primary enablement tag for this generator
	ActivationTag string

	// ValueMode declares how the ActivationTag's value is interpreted
	ValueMode ValueMode

	// GroupActivationTag is a package level enablement tag
	// whose presence activates this generator.
	// ActivationTag set to "false" always overrides this.
	GroupActivationTag string

	// ScanTypes indicates that, when ActivationTag is absent, the
	// generator scans the package's types for type-level
	// +<ActivationTag>=true tags. ActivationTag set to "false"
	// suppresses the scan and skips the package.
	ScanTypes bool

	// DefaultEnabled indicates that the generator runs on every package
	// it is asked to process unless ActivationTag is set to "false".
	DefaultEnabled bool

	// InputTag declares the tag that declares a list of input packages for this generator
	InputTag string

	// AuxTags declares other tags owned by this generator
	AuxTags []string
}

// ValueMode describes how an ActivationTag's value is interpreted.
type ValueMode int

const (
	// Boolean identifies "true"/"false" values that explicitly turn on or
	// off a generator.
	Boolean ValueMode = iota

	// Package identifies "package"/"false" tags that explicitly turn on or
	// off a generator for an entire package. Such tags typically can
	// be turned on individually for types within a package unless the value
	// is "false".
	Package

	// TypeFilterList: values are type-name patterns (e.g. "TypeMeta" for
	// defaulter-gen and validation-gen). Read via Activation.TypeFilters.
	TypeFilterList

	// ConversionPeerList is conversion-gen's bespoke tag protocol.
	// Values are either peer packages, or "false" which sets
	// the generator to explicit-only mode.
	ConversionPeerList
)

var allKnownSpecs = []Spec{
	Conversion, Defaulter, Validation, PrereleaseLifecycle,
	Deepcopy, Register, OpenAPI, Protobuf,
	Client, ApplyConfiguration, Informer, Lister,
}

// Codify the behavior of the activation tags for API definition generators.
var (
	Conversion = Spec{
		ActivationTag: "k8s:conversion-gen",
		InputTag:      "k8s:conversion-gen-external-types",
		ValueMode:     ConversionPeerList,
	}
	Defaulter = Spec{
		ActivationTag: "k8s:defaulter-gen",
		InputTag:      "k8s:defaulter-gen-input",
		ValueMode:     TypeFilterList,
	}
	Validation = Spec{
		ActivationTag: "k8s:validation-gen",
		InputTag:      "k8s:validation-gen-input",
		ValueMode:     TypeFilterList,
		AuxTags: []string{
			"k8s:validation-gen-nolint",
			"k8s:validation-gen-scheme-registry",
			"k8s:validation-gen-test-fixture",
		},
	}
	PrereleaseLifecycle = Spec{
		ActivationTag: "k8s:prerelease-lifecycle-gen",
		ValueMode:     Boolean,
	}
	Deepcopy = Spec{
		ActivationTag: "k8s:deepcopy-gen",
		ValueMode:     Package,
		ScanTypes:     true,
	}
	Register = Spec{
		ActivationTag:      "k8s:register-gen",
		ValueMode:          Boolean,
		GroupActivationTag: "groupName",
	}
	OpenAPI = Spec{
		ActivationTag: "k8s:openapi-gen",
		InputTag:      "k8s:openapi-model-package",
		ValueMode:     Boolean,
	}
	Protobuf = Spec{
		ActivationTag: "k8s:protobuf-gen",
		ValueMode:     Package,
	}
	Client = Spec{
		ActivationTag:  "k8s:client-gen",
		ValueMode:      Boolean,
		DefaultEnabled: true,
		AuxTags: []string{
			"genclient",
			"genclient:nonNamespaced",
			"genclient:noVerbs",
			"genclient:onlyVerbs",
			"genclient:skipVerbs",
			"genclient:noStatus",
			"genclient:readonly",
			"genclient:method",
			"groupGoName",
		},
	}
	ApplyConfiguration = Spec{
		ActivationTag:  "k8s:applyconfiguration-gen",
		ValueMode:      Boolean,
		DefaultEnabled: true,
	}
	Informer = Spec{
		ActivationTag:  "k8s:informer-gen",
		ValueMode:      Boolean,
		DefaultEnabled: true,
	}
	Lister = Spec{
		ActivationTag:  "k8s:lister-gen",
		ValueMode:      Boolean,
		DefaultEnabled: true,
	}
)

var allKnownTags = func() sets.Set[string] {
	out := sets.New[string]()
	for _, s := range allKnownSpecs {
		out.Insert(s.ActivationTag)
		if s.InputTag != "" {
			out.Insert(s.InputTag)
		}
		out.Insert(s.AuxTags...)
	}
	return out
}()
