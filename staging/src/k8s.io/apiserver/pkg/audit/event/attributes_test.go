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

package event

import (
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/apiserver/pkg/apis/audit"
)

func TestAttributes(t *testing.T) {
	for _, tc := range []struct {
		desc         string
		ev           *audit.Event
		path         string
		isReadOnly   bool
		resourceName string
		shouldErr    bool
	}{
		{
			desc: "has resources",
			ev: &audit.Event{
				Verb: "get",
				ObjectRef: &audit.ObjectReference{
					Resource:  "pod",
					Name:      "mypod",
					Namespace: "test",
				},
				RequestURI: "/api/v1/namespaces/test/pods",
			},
			path:         "",
			isReadOnly:   true,
			resourceName: "mypod",
			shouldErr:    false,
		},
		{
			desc: "no resources",
			ev: &audit.Event{
				Verb:       "create",
				RequestURI: "/api/v1/namespaces/test/pods",
			},
			path:         "/api/v1/namespaces/test/pods",
			isReadOnly:   false,
			resourceName: "",
			shouldErr:    false,
		},
		{
			desc: "no path or resource",
			ev: &audit.Event{
				Verb: "create",
			},
			path:         "",
			isReadOnly:   false,
			resourceName: "",
			shouldErr:    true,
		},
		{
			desc: "invalid path",
			ev: &audit.Event{
				Verb: "create",
			},
			path:         "a/bad/path",
			isReadOnly:   false,
			resourceName: "",
			shouldErr:    true,
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			attr, err := NewAttributes(tc.ev)
			if tc.shouldErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.path, attr.GetPath())
			require.Equal(t, tc.isReadOnly, attr.IsReadOnly())
			require.Equal(t, tc.resourceName, attr.GetName())
		})
	}
}
