/*
Copyright 2014 Google Inc. All rights reserved.

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

package rest

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

func TestBeforeUpdate(t *testing.T) {
	tests := []struct {
		old       runtime.Object
		obj       runtime.Object
		expectErr bool
	}{
		{
			obj: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:            "foo",
					ResourceVersion: "1",
					Namespace:       "#$%%invalid",
				},
			},
			old:       &api.Service{},
			expectErr: true,
		},
		{
			obj: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:            "foo",
					ResourceVersion: "1",
					Namespace:       "valid",
				},
			},
			old: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:            "bar",
					ResourceVersion: "1",
					Namespace:       "valid",
				},
			},
			expectErr: true,
		},
		{
			obj: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:            "foo",
					ResourceVersion: "1",
					Namespace:       "valid",
				},
				Spec: api.ServiceSpec{
					PortalIP: "1.2.3.4",
				},
			},
			old: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:            "foo",
					ResourceVersion: "1",
					Namespace:       "valid",
				},
				Spec: api.ServiceSpec{
					PortalIP: "4.3.2.1",
				},
			},
			expectErr: true,
		},
		{
			obj: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:            "foo",
					ResourceVersion: "1",
					Namespace:       api.NamespaceDefault,
				},
				Spec: api.ServiceSpec{
					PortalIP: "1.2.3.4",
					Selector: map[string]string{"foo": "bar"},
				},
			},
			old: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:            "foo",
					ResourceVersion: "1",
					Namespace:       api.NamespaceDefault,
				},
				Spec: api.ServiceSpec{
					PortalIP: "1.2.3.4",
					Selector: map[string]string{"bar": "foo"},
				},
			},
		},
	}
	for _, test := range tests {
		ctx := api.NewDefaultContext()
		err := BeforeUpdate(Services, ctx, test.obj, test.old)
		if test.expectErr && err == nil {
			t.Errorf("unexpected non-error for %v", test)
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v for %v -> %v", err, test.obj, test.old)
		}
	}
}
