/*
Copyright 2016 The Kubernetes Authors.

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

package capabilities

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

func TestGenerateAdds(t *testing.T) {
	tests := map[string]struct {
		defaultAddCaps   []api.Capability
		requiredDropCaps []api.Capability
		containerCaps    *api.Capabilities
		expectedCaps     *api.Capabilities
	}{
		"no required, no container requests": {
			expectedCaps: nil,
		},
		"required, no container requests": {
			defaultAddCaps: []api.Capability{"foo"},
			expectedCaps: &api.Capabilities{
				Add: []api.Capability{"foo"},
			},
		},
		"required, container requests add required": {
			defaultAddCaps: []api.Capability{"foo"},
			containerCaps: &api.Capabilities{
				Add: []api.Capability{"foo"},
			},
			expectedCaps: &api.Capabilities{
				Add: []api.Capability{"foo"},
			},
		},
		"multiple required, container requests add required": {
			defaultAddCaps: []api.Capability{"foo", "bar", "baz"},
			containerCaps: &api.Capabilities{
				Add: []api.Capability{"foo"},
			},
			expectedCaps: &api.Capabilities{
				Add: []api.Capability{"bar", "baz", "foo"},
			},
		},
		"required, container requests add non-required": {
			defaultAddCaps: []api.Capability{"foo"},
			containerCaps: &api.Capabilities{
				Add: []api.Capability{"bar"},
			},
			expectedCaps: &api.Capabilities{
				Add: []api.Capability{"bar", "foo"},
			},
		},
		"generation dedupes": {
			defaultAddCaps: []api.Capability{"foo", "foo", "foo", "foo"},
			containerCaps: &api.Capabilities{
				Add: []api.Capability{"foo", "foo", "foo"},
			},
			expectedCaps: &api.Capabilities{
				Add: []api.Capability{"foo"},
			},
		},
		"generation is case sensitive - will not dedupe": {
			defaultAddCaps: []api.Capability{"foo"},
			containerCaps: &api.Capabilities{
				Add: []api.Capability{"FOO"},
			},
			expectedCaps: &api.Capabilities{
				Add: []api.Capability{"FOO", "foo"},
			},
		},
	}

	for k, v := range tests {
		container := &api.Container{
			SecurityContext: &api.SecurityContext{
				Capabilities: v.containerCaps,
			},
		}

		strategy, err := NewDefaultCapabilities(v.defaultAddCaps, v.requiredDropCaps, nil)
		if err != nil {
			t.Errorf("%s failed: %v", k, err)
			continue
		}
		generatedCaps, err := strategy.Generate(nil, container)
		if err != nil {
			t.Errorf("%s failed generating: %v", k, err)
			continue
		}
		if v.expectedCaps == nil && generatedCaps != nil {
			t.Errorf("%s expected nil caps to be generated but got %v", k, generatedCaps)
			continue
		}
		if !reflect.DeepEqual(v.expectedCaps, generatedCaps) {
			t.Errorf("%s did not generate correctly.  Expected: %#v, Actual: %#v", k, v.expectedCaps, generatedCaps)
		}
	}
}

func TestGenerateDrops(t *testing.T) {
	tests := map[string]struct {
		defaultAddCaps   []api.Capability
		requiredDropCaps []api.Capability
		containerCaps    *api.Capabilities
		expectedCaps     *api.Capabilities
	}{
		"no required, no container requests": {
			expectedCaps: nil,
		},
		"required drops are defaulted": {
			requiredDropCaps: []api.Capability{"foo"},
			expectedCaps: &api.Capabilities{
				Drop: []api.Capability{"foo"},
			},
		},
		"required drops are defaulted when making container requests": {
			requiredDropCaps: []api.Capability{"foo"},
			containerCaps: &api.Capabilities{
				Drop: []api.Capability{"foo", "bar"},
			},
			expectedCaps: &api.Capabilities{
				Drop: []api.Capability{"bar", "foo"},
			},
		},
		"can drop a required add": {
			defaultAddCaps: []api.Capability{"foo"},
			containerCaps: &api.Capabilities{
				Drop: []api.Capability{"foo"},
			},
			expectedCaps: &api.Capabilities{
				Drop: []api.Capability{"foo"},
			},
		},
		"can drop non-required add": {
			defaultAddCaps: []api.Capability{"foo"},
			containerCaps: &api.Capabilities{
				Drop: []api.Capability{"bar"},
			},
			expectedCaps: &api.Capabilities{
				Add:  []api.Capability{"foo"},
				Drop: []api.Capability{"bar"},
			},
		},
		"defaulting adds and drops, dropping a required add": {
			defaultAddCaps:   []api.Capability{"foo", "bar", "baz"},
			requiredDropCaps: []api.Capability{"abc"},
			containerCaps: &api.Capabilities{
				Drop: []api.Capability{"foo"},
			},
			expectedCaps: &api.Capabilities{
				Add:  []api.Capability{"bar", "baz"},
				Drop: []api.Capability{"abc", "foo"},
			},
		},
		"generation dedupes": {
			requiredDropCaps: []api.Capability{"bar", "bar", "bar", "bar"},
			containerCaps: &api.Capabilities{
				Drop: []api.Capability{"bar", "bar", "bar"},
			},
			expectedCaps: &api.Capabilities{
				Drop: []api.Capability{"bar"},
			},
		},
		"generation is case sensitive - will not dedupe": {
			requiredDropCaps: []api.Capability{"bar"},
			containerCaps: &api.Capabilities{
				Drop: []api.Capability{"BAR"},
			},
			expectedCaps: &api.Capabilities{
				Drop: []api.Capability{"BAR", "bar"},
			},
		},
	}
	for k, v := range tests {
		container := &api.Container{
			SecurityContext: &api.SecurityContext{
				Capabilities: v.containerCaps,
			},
		}

		strategy, err := NewDefaultCapabilities(v.defaultAddCaps, v.requiredDropCaps, nil)
		if err != nil {
			t.Errorf("%s failed: %v", k, err)
			continue
		}
		generatedCaps, err := strategy.Generate(nil, container)
		if err != nil {
			t.Errorf("%s failed generating: %v", k, err)
			continue
		}
		if v.expectedCaps == nil && generatedCaps != nil {
			t.Errorf("%s expected nil caps to be generated but got %#v", k, generatedCaps)
			continue
		}
		if !reflect.DeepEqual(v.expectedCaps, generatedCaps) {
			t.Errorf("%s did not generate correctly.  Expected: %#v, Actual: %#v", k, v.expectedCaps, generatedCaps)
		}
	}
}

func TestValidateAdds(t *testing.T) {
	tests := map[string]struct {
		defaultAddCaps   []api.Capability
		requiredDropCaps []api.Capability
		allowedCaps      []api.Capability
		containerCaps    *api.Capabilities
		shouldPass       bool
	}{
		// no container requests
		"no required, no allowed, no container requests": {
			shouldPass: true,
		},
		"no required, allowed, no container requests": {
			allowedCaps: []api.Capability{"foo"},
			shouldPass:  true,
		},
		"required, no allowed, no container requests": {
			defaultAddCaps: []api.Capability{"foo"},
			shouldPass:     false,
		},

		// container requests match required
		"required, no allowed, container requests valid": {
			defaultAddCaps: []api.Capability{"foo"},
			containerCaps: &api.Capabilities{
				Add: []api.Capability{"foo"},
			},
			shouldPass: true,
		},
		"required, no allowed, container requests invalid": {
			defaultAddCaps: []api.Capability{"foo"},
			containerCaps: &api.Capabilities{
				Add: []api.Capability{"bar"},
			},
			shouldPass: false,
		},

		// container requests match allowed
		"no required, allowed, container requests valid": {
			allowedCaps: []api.Capability{"foo"},
			containerCaps: &api.Capabilities{
				Add: []api.Capability{"foo"},
			},
			shouldPass: true,
		},
		"no required, allowed, container requests invalid": {
			allowedCaps: []api.Capability{"foo"},
			containerCaps: &api.Capabilities{
				Add: []api.Capability{"bar"},
			},
			shouldPass: false,
		},

		// required and allowed
		"required, allowed, container requests valid required": {
			defaultAddCaps: []api.Capability{"foo"},
			allowedCaps:    []api.Capability{"bar"},
			containerCaps: &api.Capabilities{
				Add: []api.Capability{"foo"},
			},
			shouldPass: true,
		},
		"required, allowed, container requests valid allowed": {
			defaultAddCaps: []api.Capability{"foo"},
			allowedCaps:    []api.Capability{"bar"},
			containerCaps: &api.Capabilities{
				Add: []api.Capability{"bar"},
			},
			shouldPass: true,
		},
		"required, allowed, container requests invalid": {
			defaultAddCaps: []api.Capability{"foo"},
			allowedCaps:    []api.Capability{"bar"},
			containerCaps: &api.Capabilities{
				Add: []api.Capability{"baz"},
			},
			shouldPass: false,
		},
		"validation is case sensitive": {
			defaultAddCaps: []api.Capability{"foo"},
			containerCaps: &api.Capabilities{
				Add: []api.Capability{"FOO"},
			},
			shouldPass: false,
		},
	}

	for k, v := range tests {
		container := &api.Container{
			SecurityContext: &api.SecurityContext{
				Capabilities: v.containerCaps,
			},
		}

		strategy, err := NewDefaultCapabilities(v.defaultAddCaps, v.requiredDropCaps, v.allowedCaps)
		if err != nil {
			t.Errorf("%s failed: %v", k, err)
			continue
		}
		errs := strategy.Validate(nil, container)
		if v.shouldPass && len(errs) > 0 {
			t.Errorf("%s should have passed but had errors %v", k, errs)
			continue
		}
		if !v.shouldPass && len(errs) == 0 {
			t.Errorf("%s should have failed but recieved no errors", k)
		}
	}
}

func TestValidateDrops(t *testing.T) {
	tests := map[string]struct {
		defaultAddCaps   []api.Capability
		requiredDropCaps []api.Capability
		containerCaps    *api.Capabilities
		shouldPass       bool
	}{
		// no container requests
		"no required, no container requests": {
			shouldPass: true,
		},
		"required, no container requests": {
			requiredDropCaps: []api.Capability{"foo"},
			shouldPass:       false,
		},

		// container requests match required
		"required, container requests valid": {
			requiredDropCaps: []api.Capability{"foo"},
			containerCaps: &api.Capabilities{
				Drop: []api.Capability{"foo"},
			},
			shouldPass: true,
		},
		"required, container requests invalid": {
			requiredDropCaps: []api.Capability{"foo"},
			containerCaps: &api.Capabilities{
				Drop: []api.Capability{"bar"},
			},
			shouldPass: false,
		},
		"validation is case sensitive": {
			requiredDropCaps: []api.Capability{"foo"},
			containerCaps: &api.Capabilities{
				Drop: []api.Capability{"FOO"},
			},
			shouldPass: false,
		},
	}

	for k, v := range tests {
		container := &api.Container{
			SecurityContext: &api.SecurityContext{
				Capabilities: v.containerCaps,
			},
		}

		strategy, err := NewDefaultCapabilities(v.defaultAddCaps, v.requiredDropCaps, nil)
		if err != nil {
			t.Errorf("%s failed: %v", k, err)
			continue
		}
		errs := strategy.Validate(nil, container)
		if v.shouldPass && len(errs) > 0 {
			t.Errorf("%s should have passed but had errors %v", k, errs)
			continue
		}
		if !v.shouldPass && len(errs) == 0 {
			t.Errorf("%s should have failed but recieved no errors", k)
		}
	}
}
