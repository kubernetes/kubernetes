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

package enforced

import (
	"testing"

	"github.com/stretchr/testify/require"

	authnv1 "k8s.io/api/authentication/v1"
	"k8s.io/apimachinery/pkg/runtime"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit/policy"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	fakeplugin "k8s.io/apiserver/plugin/pkg/audit/fake"
)

func TestEnforced(t *testing.T) {
	testCases := []struct {
		name     string
		event    *auditinternal.Event
		policy   auditinternal.Policy
		attribs  authorizer.Attributes
		expected []*auditinternal.Event
	}{
		{
			name: "enforce level",
			event: &auditinternal.Event{
				Level:          auditinternal.LevelRequestResponse,
				Stage:          auditinternal.StageResponseComplete,
				RequestURI:     "/apis/extensions/v1beta1",
				RequestObject:  &runtime.Unknown{Raw: []byte(`test`)},
				ResponseObject: &runtime.Unknown{Raw: []byte(`test`)},
			},
			policy: auditinternal.Policy{
				Rules: []auditinternal.PolicyRule{
					{
						Level: auditinternal.LevelMetadata,
					},
				},
			},
			expected: []*auditinternal.Event{
				{
					Level:      auditinternal.LevelMetadata,
					Stage:      auditinternal.StageResponseComplete,
					RequestURI: "/apis/extensions/v1beta1",
				},
			},
		},
		{
			name: "enforce policy rule",
			event: &auditinternal.Event{
				Level:      auditinternal.LevelRequestResponse,
				Stage:      auditinternal.StageResponseComplete,
				RequestURI: "/apis/extensions/v1beta1",
				User: authnv1.UserInfo{
					Username: user.Anonymous,
				},
				RequestObject:  &runtime.Unknown{Raw: []byte(`test`)},
				ResponseObject: &runtime.Unknown{Raw: []byte(`test`)},
			},
			policy: auditinternal.Policy{
				Rules: []auditinternal.PolicyRule{
					{
						Level: auditinternal.LevelNone,
						Users: []string{user.Anonymous},
					},
					{
						Level: auditinternal.LevelMetadata,
					},
				},
			},
			expected: []*auditinternal.Event{},
		},
		{
			name:  "nil event",
			event: nil,
			policy: auditinternal.Policy{
				Rules: []auditinternal.PolicyRule{
					{
						Level: auditinternal.LevelMetadata,
					},
				},
			},
			expected: []*auditinternal.Event{},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ev := []*auditinternal.Event{}
			fakeBackend := fakeplugin.Backend{
				OnRequest: func(events []*auditinternal.Event) {
					ev = events
				},
			}
			b := NewBackend(&fakeBackend, policy.NewChecker(&tc.policy))
			defer b.Shutdown()

			b.ProcessEvents(tc.event)
			require.Equal(t, tc.expected, ev)
		})
	}
}
