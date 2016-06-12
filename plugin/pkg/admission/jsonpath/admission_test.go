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
		ops = []admission.Operation{admission.Update, admission.Delete, admission.Connect}
	}
	result := []admission.Attributes{}
	for _, op := range ops {
		result = append(result, admission.NewAttributesRecord(obj, nil, api.Kind(kind).WithVersion("version"), "myns", "myname", api.Resource(resource).WithVersion("version"), "", op, nil))
	}
	return result
}

func TestJSONPathMultiRule(t *testing.T) {
	config := JSONPathAdmissionConfig{
		Rules: []JSONPathAdmissionRule{
			JSONPathAdmissionRule{
				KindRegexp:  "Pod",
				Path:        ".metadata.name",
				MatchRegexp: "^Foo.*$",
			},
			JSONPathAdmissionRule{
				KindRegexp:  "Service",
				Path:        ".metadata.name",
				MatchRegexp: "^Foo.*$",
			},
			JSONPathAdmissionRule{
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
						api.Container{
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
		attrs := makeAttributes(test.kind, test.rsrc, test.obj, nil)
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
						api.Container{
							Name: "FooBar",
						},
						api.Container{
							Name: "FooBaz",
						},
					},
				},
			},
		},
		{
			name: "multi-match fail",
			rule: JSONPathAdmissionRule{
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
						api.Container{
							Name: "FooBar",
						},
						api.Container{
							Name: "FooBaz",
						},
						api.Container{
							Name: "BazFooBar",
						},
					},
				},
			},
		},
		{
			name: "multi-match no match fail",
			rule: JSONPathAdmissionRule{
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
