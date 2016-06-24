/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package jsonpath

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"testing"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
)

func makeAttributes(kind, resource string, obj runtime.Object, ops []admission.Operation) []admission.Attributes {
	if ops == nil {
		ops = []admission.Operation{admission.Create, admission.Update}
	}
	result := []admission.Attributes{}
	for _, op := range ops {
		result = append(result, admission.NewAttributesRecord(obj, nil, api.Kind(kind).WithVersion("version"), "myns", "myname", api.Resource(resource).WithVersion("version"), "", op, nil))
	}
	return result
}

func TestHandles(t *testing.T) {
	j, err := NewJSONPathAdmission(JSONPathAdmissionConfig{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	handles := []admission.Operation{admission.Create, admission.Update}
	for _, op := range handles {
		if !j.Handles(op) {
			t.Errorf("expected to handle: %s but didn't", op)
		}
	}

	doesntHandle := []admission.Operation{admission.Connect, admission.Delete}
	for _, op := range doesntHandle {
		if j.Handles(op) {
			t.Errorf("expected not to handle: %s, but did", op)
		}
	}
}

func TestJSONPathMultiRule(t *testing.T) {
	config := JSONPathAdmissionConfig{
		Rules: []JSONPathAdmissionRule{
			{
				APIVersion:  "v1",
				KindRegexp:  "Pod",
				Path:        ".metadata.name",
				MatchRegexp: "^Foo.*$",
			},
			{
				APIVersion:  "v1",
				KindRegexp:  "Service",
				Path:        ".metadata.name",
				MatchRegexp: "^Foo.*$",
			},
			{
				APIVersion:  "v1",
				KindRegexp:  "Pod",
				Path:        ".spec.containers[*].name",
				MatchRegexp: "^Foo.*$",
			},
		},
	}
	file, err := ioutil.TempFile(os.TempDir(), "config")
	if err != nil {
		t.Errorf("unexpected error making tmp file: %v", err)
		return
	}
	defer os.Remove(file.Name())
	data, err := json.Marshal(config)
	if err != nil {
		t.Errorf("unexpected error marshaling data: %v", err)
		return
	}
	fmt.Fprintf(file, string(data))
	if err := file.Close(); err != nil {
		t.Errorf("unexpected error closing file: %v", err)
	}

	a := admission.InitPlugin("jsonpath", nil, file.Name())
	if a == nil {
		t.Errorf("unexpected nil plugin")
		return
	}

	tests := []struct {
		name  string
		kind  string
		rsrc  string
		obj   runtime.Object
		admit bool
		ops   []admission.Operation
	}{
		{
			name:  "empty",
			kind:  "Pod",
			rsrc:  "pods",
			obj:   &api.Pod{},
			admit: false,
		},
		{
			name: "one condition",
			kind: "Pod",
			rsrc: "pods",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "FooBar",
				},
			},
			admit: false,
		},
		{
			name: "all condition",
			kind: "Pod",
			rsrc: "pods",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "FooBar",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "FooBaz",
						},
					},
				},
			},
			admit: true,
		},
		{
			name: "other kind",
			kind: "Service",
			rsrc: "services",
			obj: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name: "FooBar",
				},
			},
			admit: true,
		},
		{
			name:  "other kind empty",
			kind:  "Service",
			rsrc:  "services",
			obj:   &api.Service{},
			admit: false,
		},
	}
	for _, test := range tests {
		attrs := makeAttributes(test.kind, test.rsrc, test.obj, test.ops)
		for _, attr := range attrs {
			err := a.Admit(attr)
			if test.admit && err != nil {
				t.Errorf("expected no error, saw: %v for %s", err, test.name)
				continue
			}
			if !test.admit && err == nil {
				t.Errorf("unexpected non-error for %s", test.name)
			}
		}
	}

}

func TestJSONPathRule(t *testing.T) {
	tests := []struct {
		name  string
		rule  JSONPathAdmissionRule
		admit bool
		kind  string
		rsrc  string
		obj   runtime.Object
		ops   []admission.Operation
	}{
		{
			name: "simple",
			rule: JSONPathAdmissionRule{
				APIVersion:  "v1",
				KindRegexp:  "Pod",
				Path:        ".metadata.name",
				MatchRegexp: "^Foo.*$",
			},
			admit: true,
			kind:  "Pod",
			rsrc:  "pods",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "FooBar",
				},
			},
		},
		{
			name: "simple fail",
			rule: JSONPathAdmissionRule{
				APIVersion:  "v1",
				KindRegexp:  "Pod",
				Path:        ".metadata.name",
				MatchRegexp: "^Foo.*$",
			},
			admit: false,
			kind:  "Pod",
			rsrc:  "pods",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "BazFooBar",
				},
			},
		},
		{
			name: "no kind match",
			rule: JSONPathAdmissionRule{
				APIVersion:  "v1",
				KindRegexp:  "Pod",
				Path:        ".metadata.name",
				MatchRegexp: "^Foo.*$",
			},
			admit: true,
			kind:  "Service",
			rsrc:  "services",
			obj: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name: "BazFooBar",
				},
			},
		},
		{
			name: "complicated kind-match",
			rule: JSONPathAdmissionRule{
				APIVersion:  "v1",
				KindRegexp:  "(Service)|(Pod)",
				Path:        ".metadata.name",
				MatchRegexp: "^Foo.*$",
			},
			admit: true,
			kind:  "Pod",
			rsrc:  "pods",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "FooBar",
				},
			},
		},
		{
			name: "complicated kind-match",
			rule: JSONPathAdmissionRule{
				APIVersion:  "v1",
				KindRegexp:  "(Service)|(Pod)",
				Path:        ".metadata.name",
				MatchRegexp: "^Foo.*$",
			},
			admit: false,
			kind:  "Service",
			rsrc:  "services",
			obj: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name: "BazFooBar",
				},
			},
		},
		{
			name: "multi-match",
			rule: JSONPathAdmissionRule{
				APIVersion:  "v1",
				KindRegexp:  "Pod",
				Path:        ".spec.containers[*].name",
				MatchRegexp: "^Foo.*$",
			},
			admit: true,
			kind:  "Pod",
			rsrc:  "pods",
			obj: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "FooBar",
						},
						{
							Name: "FooBaz",
						},
					},
				},
			},
		},
		{
			name: "multi-match fail",
			rule: JSONPathAdmissionRule{
				APIVersion:  "v1",
				KindRegexp:  "Pod",
				Path:        ".spec.containers[*].name",
				MatchRegexp: "^Foo.*$",
			},
			admit: false,
			kind:  "Pod",
			rsrc:  "pods",
			obj: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "FooBar",
						},
						{
							Name: "FooBaz",
						},
						{
							Name: "BazFooBar",
						},
					},
				},
			},
		},
		{
			name: "multi-match no match fail",
			rule: JSONPathAdmissionRule{
				APIVersion:  "v1",
				KindRegexp:  "Pod",
				Path:        ".spec.containers[*].name",
				MatchRegexp: "^Foo.*$",
			},
			admit: false,
			kind:  "Pod",
			rsrc:  "pods",
			obj: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{},
				},
			},
		},
		{
			name: "multi-match no match ok",
			rule: JSONPathAdmissionRule{
				APIVersion:       "v1",
				KindRegexp:       "Pod",
				Path:             ".spec.containers[*].name",
				MatchRegexp:      "^Foo.*$",
				AcceptEmptyMatch: true,
			},
			admit: true,
			kind:  "Pod",
			rsrc:  "pods",
			obj: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{},
				},
			},
		},
	}

	for _, test := range tests {
		attrs := makeAttributes(test.kind, test.rsrc, test.obj, test.ops)
		for _, attr := range attrs {
			rule, err := makeRule(&test.rule)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				continue
			}
			err = rule.admit(attr)
			if test.admit && err != nil {
				t.Errorf("Expected nil, got %v for %s", err, test.name)
				continue
			}
			if !test.admit && err == nil {
				t.Errorf("Unexpected nil for %s", test.name)
			}
		}
	}
}
