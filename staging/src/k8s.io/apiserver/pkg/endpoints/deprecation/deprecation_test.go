/*
Copyright 2020 The Kubernetes Authors.

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

package deprecation

import (
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/version"
)

func TestMajorMinor(t *testing.T) {
	tests := []struct {
		name        string
		v           version.Info
		expectMajor int
		expectMinor int
		expectErr   bool
	}{
		{
			name:        "empty",
			v:           version.Info{Major: "", Minor: ""},
			expectMajor: 0,
			expectMinor: 0,
			expectErr:   true,
		},
		{
			name:        "non-numeric major",
			v:           version.Info{Major: "A", Minor: "0"},
			expectMajor: 0,
			expectMinor: 0,
			expectErr:   true,
		},
		{
			name:        "non-numeric minor",
			v:           version.Info{Major: "1", Minor: "A"},
			expectMajor: 0,
			expectMinor: 0,
			expectErr:   true,
		},
		{
			name:        "valid",
			v:           version.Info{Major: "1", Minor: "2"},
			expectMajor: 1,
			expectMinor: 2,
			expectErr:   false,
		},
		{
			name:        "valid zero",
			v:           version.Info{Major: "0", Minor: "0"},
			expectMajor: 0,
			expectMinor: 0,
			expectErr:   false,
		},
		{
			name:        "valid zero decimal",
			v:           version.Info{Major: "01", Minor: "02"},
			expectMajor: 1,
			expectMinor: 2,
			expectErr:   false,
		},
		{
			name:        "valid with extra minor",
			v:           version.Info{Major: "1", Minor: "2+"},
			expectMajor: 1,
			expectMinor: 2,
			expectErr:   false,
		},
		{
			name:        "valid with extra minor",
			v:           version.Info{Major: "1", Minor: "2.3"},
			expectMajor: 1,
			expectMinor: 2,
			expectErr:   false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			major, minor, err := MajorMinor(tt.v)
			if (err != nil) != tt.expectErr {
				t.Errorf("MajorMinor() error = %v, wantErr %v", err, tt.expectErr)
				return
			}
			if major != tt.expectMajor {
				t.Errorf("MajorMinor() major = %v, want %v", major, tt.expectMajor)
			}
			if minor != tt.expectMinor {
				t.Errorf("MajorMinor() minor = %v, want %v", minor, tt.expectMinor)
			}
		})
	}
}

func TestIsDeprecated(t *testing.T) {
	tests := []struct {
		name string

		obj          runtime.Object
		currentMajor int
		currentMinor int

		want bool
	}{
		{
			name:         "no interface",
			obj:          &fakeObject{},
			currentMajor: 0,
			currentMinor: 0,
			want:         false,
		},
		{
			name:         "interface, zero-value",
			obj:          &fakeDeprecatedObject{},
			currentMajor: 0,
			currentMinor: 0,
			want:         false,
		},
		{
			name:         "interface, non-zero-value, no current value",
			obj:          &fakeDeprecatedObject{major: 10, minor: 20},
			currentMajor: 0,
			currentMinor: 0,
			want:         true,
		},
		{
			name:         "interface, non-zero-value matching major, minor",
			obj:          &fakeDeprecatedObject{major: 10, minor: 20},
			currentMajor: 10,
			currentMinor: 20,
			want:         true,
		},
		{
			name:         "interface, non-zero-value after major, after minor",
			obj:          &fakeDeprecatedObject{major: 10, minor: 20},
			currentMajor: 9,
			currentMinor: 19,
			want:         false,
		},
		{
			name:         "interface, non-zero-value after major, before minor",
			obj:          &fakeDeprecatedObject{major: 10, minor: 20},
			currentMajor: 9,
			currentMinor: 21,
			want:         false,
		},
		{
			name:         "interface, non-zero-value before major, after minor",
			obj:          &fakeDeprecatedObject{major: 10, minor: 20},
			currentMajor: 11,
			currentMinor: 19,
			want:         true,
		},
		{
			name:         "interface, non-zero-value before major, before minor",
			obj:          &fakeDeprecatedObject{major: 10, minor: 20},
			currentMajor: 11,
			currentMinor: 21,
			want:         true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsDeprecated(tt.obj, tt.currentMajor, tt.currentMinor); got != tt.want {
				t.Errorf("IsDeprecated() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRemovedRelease(t *testing.T) {
	tests := []struct {
		name string
		obj  runtime.Object
		want string
	}{
		{
			name: "no interface",
			obj:  &fakeObject{},
			want: "",
		},
		{
			name: "interface, zero-value",
			obj:  &fakeRemovedObject{removedMajor: 0, removedMinor: 0},
			want: "",
		},
		{
			name: "interface, non-zero major",
			obj:  &fakeRemovedObject{removedMajor: 1, removedMinor: 0},
			want: "1.0",
		},
		{
			name: "interface, non-zero minor",
			obj:  &fakeRemovedObject{removedMajor: 0, removedMinor: 1},
			want: "0.1",
		},
		{
			name: "interface, non-zero",
			obj:  &fakeRemovedObject{removedMajor: 1, removedMinor: 2},
			want: "1.2",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := RemovedRelease(tt.obj); got != tt.want {
				t.Errorf("RemovedRelease() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestWarningMessage(t *testing.T) {
	tests := []struct {
		name string
		obj  runtime.Object
		gvk  schema.GroupVersionKind
		want string
	}{
		{
			name: "no interface, zero-value",
			obj:  &fakeObject{},
			want: "",
		},
		{
			name: "deprecated interface, zero-value",
			obj:  &fakeDeprecatedObject{major: 0, minor: 0},
			gvk:  schema.GroupVersionKind{},
			want: "",
		},
		{
			name: "deprecated interface, non-zero-value",
			obj:  &fakeDeprecatedObject{major: 1, minor: 2},
			gvk:  schema.GroupVersionKind{Group: "mygroup", Version: "v1", Kind: "MyKind"},
			want: "mygroup/v1 MyKind is deprecated in v1.2+",
		},
		{
			name: "removed interface, zero-value removal version",
			obj:  &fakeRemovedObject{major: 1, minor: 2},
			gvk:  schema.GroupVersionKind{Group: "mygroup", Version: "v1", Kind: "MyKind"},
			want: "mygroup/v1 MyKind is deprecated in v1.2+",
		},
		{
			name: "removed interface, non-zero-value removal version",
			obj:  &fakeRemovedObject{major: 1, minor: 2, removedMajor: 3, removedMinor: 4},
			gvk:  schema.GroupVersionKind{Group: "mygroup", Version: "v1", Kind: "MyKind"},
			want: "mygroup/v1 MyKind is deprecated in v1.2+, unavailable in v3.4+",
		},
		{
			name: "replaced interface, zero-value replacement",
			obj:  &fakeReplacedObject{major: 1, minor: 2, removedMajor: 3, removedMinor: 4},
			gvk:  schema.GroupVersionKind{Group: "mygroup", Version: "v1", Kind: "MyKind"},
			want: "mygroup/v1 MyKind is deprecated in v1.2+, unavailable in v3.4+",
		},
		{
			name: "replaced interface, non-zero-value replacement",
			obj:  &fakeReplacedObject{major: 1, minor: 2, removedMajor: 3, removedMinor: 4, replacement: schema.GroupVersionKind{Group: "anothergroup", Version: "v2", Kind: "AnotherKind"}},
			gvk:  schema.GroupVersionKind{Group: "mygroup", Version: "v1", Kind: "MyKind"},
			want: "mygroup/v1 MyKind is deprecated in v1.2+, unavailable in v3.4+; use anothergroup/v2 AnotherKind",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.obj.GetObjectKind().SetGroupVersionKind(tt.gvk)
			if got := WarningMessage(tt.obj); got != tt.want {
				t.Errorf("WarningMessage() = %v, want %v", got, tt.want)
			}
		})
	}
}

type fakeObject struct {
	unstructured.Unstructured
}
type fakeDeprecatedObject struct {
	unstructured.Unstructured
	major int
	minor int
}

func (f *fakeDeprecatedObject) APILifecycleDeprecated() (int, int) { return f.major, f.minor }

type fakeRemovedObject struct {
	unstructured.Unstructured
	major        int
	minor        int
	removedMajor int
	removedMinor int
}

func (f *fakeRemovedObject) APILifecycleDeprecated() (int, int) { return f.major, f.minor }
func (f *fakeRemovedObject) APILifecycleRemoved() (int, int)    { return f.removedMajor, f.removedMinor }

type fakeReplacedObject struct {
	unstructured.Unstructured
	major        int
	minor        int
	replacement  schema.GroupVersionKind
	removedMajor int
	removedMinor int
}

func (f *fakeReplacedObject) APILifecycleDeprecated() (int, int)               { return f.major, f.minor }
func (f *fakeReplacedObject) APILifecycleRemoved() (int, int)                  { return f.removedMajor, f.removedMinor }
func (f *fakeReplacedObject) APILifecycleReplacement() schema.GroupVersionKind { return f.replacement }
