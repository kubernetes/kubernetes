// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ext

import (
	"fmt"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/env"
)

// ExtensionOptionFactory converts an ExtensionConfig value to a CEL environment option.
func ExtensionOptionFactory(configElement any) (cel.EnvOption, bool) {
	ext, isExtension := configElement.(*env.Extension)
	if !isExtension {
		return nil, false
	}
	fac, found := extFactories[ext.Name]
	if !found {
		return nil, false
	}
	// If the version is 'latest', set the version value to the max uint.
	ver, err := ext.VersionNumber()
	if err != nil {
		return func(*cel.Env) (*cel.Env, error) {
			return nil, fmt.Errorf("invalid extension version: %s - %s", ext.Name, ext.Version)
		}, true
	}
	return fac(ver), true
}

// extensionFactory accepts a version and produces a CEL environment associated with the versioned extension.
type extensionFactory func(uint32) cel.EnvOption

var extFactories = map[string]extensionFactory{
	"bindings": func(version uint32) cel.EnvOption {
		return Bindings(BindingsVersion(version))
	},
	"encoders": func(version uint32) cel.EnvOption {
		return Encoders(EncodersVersion(version))
	},
	"lists": func(version uint32) cel.EnvOption {
		return Lists(ListsVersion(version))
	},
	"math": func(version uint32) cel.EnvOption {
		return Math(MathVersion(version))
	},
	"protos": func(version uint32) cel.EnvOption {
		return Protos(ProtosVersion(version))
	},
	"sets": func(version uint32) cel.EnvOption {
		return Sets(SetsVersion(version))
	},
	"strings": func(version uint32) cel.EnvOption {
		return Strings(StringsVersion(version))
	},
	"two-var-comprehensions": func(version uint32) cel.EnvOption {
		return TwoVarComprehensions(TwoVarComprehensionsVersion(version))
	},
}
