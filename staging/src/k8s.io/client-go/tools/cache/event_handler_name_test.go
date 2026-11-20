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

package cache

import (
	"reflect"
	"testing"
)

type mockHandler struct{}

func (m mockHandler) OnAdd(any, bool)   {}
func (m mockHandler) OnUpdate(any, any) {}
func (m mockHandler) OnDelete(any)      {}

func TestNameForHandler(t *testing.T) {
	pkgPrefix := reflect.TypeFor[mockHandler]().PkgPath() + "."
	emptyHandler := ResourceEventHandlerFuncs{}

	for name, tc := range map[string]struct {
		handler  ResourceEventHandler
		wantName string
	}{
		"mixture": {
			handler: ResourceEventHandlerFuncs{
				UpdateFunc: emptyHandler.OnUpdate,
				DeleteFunc: func(any) {},
			},
			wantName: pkgPrefix + "ResourceEventHandlerFuncs.OnUpdate-fm+" + pkgPrefix + "TestNameForHandler.func1", // Testcase must come first to get func1.
		},
		"add": {
			handler:  ResourceEventHandlerFuncs{AddFunc: func(any) {}},
			wantName: pkgPrefix + "TestNameForHandler",
		},
		"update": {
			handler:  ResourceEventHandlerFuncs{UpdateFunc: func(any, any) {}},
			wantName: pkgPrefix + "TestNameForHandler",
		},
		"delete": {
			handler:  ResourceEventHandlerFuncs{DeleteFunc: func(any) {}},
			wantName: pkgPrefix + "TestNameForHandler",
		},
		"all": {
			handler: ResourceEventHandlerFuncs{
				AddFunc:    func(any) {},
				UpdateFunc: func(any, any) {},
				DeleteFunc: func(any) {},
			},
			wantName: pkgPrefix + "TestNameForHandler",
		},
		"ptrToFuncs": {
			handler:  &ResourceEventHandlerFuncs{AddFunc: func(any) {}},
			wantName: pkgPrefix + "TestNameForHandler",
		},
		"struct": {
			handler:  mockHandler{},
			wantName: pkgPrefix + "mockHandler",
		},
		"ptrToStruct": {
			handler:  &mockHandler{},
			wantName: pkgPrefix + "mockHandler",
		},
		"nil": {
			handler:  nil,
			wantName: "<nil>",
		},
		"stored-nil": {
			// This is a bit odd, but one unit test actually registered
			// such an event handler and it somehow worked.
			handler:  (*mockHandler)(nil),
			wantName: "*cache.mockHandler",
		},
	} {
		t.Run(name, func(t *testing.T) {
			gotName := nameForHandler(tc.handler)
			if gotName != tc.wantName {
				t.Errorf("Got name:\n    %s\nWanted name:\n    %s", gotName, tc.wantName)
			}
		})
	}
}
