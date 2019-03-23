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

package sysctl

import (
	"testing"

	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestValidate(t *testing.T) {
	tests := map[string]struct {
		whitelist     []string
		forbiddenSafe []string
		allowedUnsafe []string
		allowed       []string
		disallowed    []string
	}{
		// no container requests
		"with allow all": {
			whitelist: []string{"foo"},
			allowed:   []string{"foo"},
		},
		"empty": {
			whitelist:     []string{"foo"},
			forbiddenSafe: []string{"*"},
			disallowed:    []string{"foo"},
		},
		"without wildcard": {
			whitelist:  []string{"a", "a.b"},
			allowed:    []string{"a", "a.b"},
			disallowed: []string{"b"},
		},
		"with catch-all wildcard and non-wildcard": {
			allowedUnsafe: []string{"a.b.c", "*"},
			allowed:       []string{"a", "a.b", "a.b.c", "b"},
		},
		"without catch-all wildcard": {
			allowedUnsafe: []string{"a.*", "b.*", "c.d.e", "d.e.f.*"},
			allowed:       []string{"a.b", "b.c", "c.d.e", "d.e.f.g.h"},
			disallowed:    []string{"a", "b", "c", "c.d", "d.e", "d.e.f"},
		},
	}

	for k, v := range tests {
		strategy := NewMustMatchPatterns(v.whitelist, v.allowedUnsafe, v.forbiddenSafe)

		pod := &api.Pod{}
		errs := strategy.Validate(pod)
		if len(errs) != 0 {
			t.Errorf("%s: unexpected validaton errors for empty sysctls: %v", k, errs)
		}

		testAllowed := func() {
			sysctls := []api.Sysctl{}
			for _, s := range v.allowed {
				sysctls = append(sysctls, api.Sysctl{
					Name:  s,
					Value: "dummy",
				})
			}
			pod.Spec.SecurityContext = &api.PodSecurityContext{
				Sysctls: sysctls,
			}
			errs = strategy.Validate(pod)
			if len(errs) != 0 {
				t.Errorf("%s: unexpected validaton errors for sysctls: %v", k, errs)
			}
		}
		testDisallowed := func() {
			for _, s := range v.disallowed {
				pod.Spec.SecurityContext = &api.PodSecurityContext{
					Sysctls: []api.Sysctl{
						{
							Name:  s,
							Value: "dummy",
						},
					},
				}
				errs = strategy.Validate(pod)
				if len(errs) == 0 {
					t.Errorf("%s: expected error for sysctl %q", k, s)
				}
			}
		}

		testAllowed()
		testDisallowed()
	}
}
