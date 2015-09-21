package conversion

import (
	"encoding/json"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
)

type Foo struct {
	unversioned.ListMeta `json:"metadata"`
}

func TestSpliceResourceVersion(t *testing.T) {
	foo := Foo{
		ListMeta: unversioned.ListMeta{
			SelfLink: "http://foo/bar",
		},
	}

	data, err := json.Marshal(foo)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	output, err := spliceResourceVersion(string(data), "22")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	fooOut := Foo{}
	if err := json.Unmarshal(output, &fooOut); err != nil {
		t.Errorf("unexpected error: %v for '%s'", err, string(output))
	}

	foo.ResourceVersion = "22"
	if !reflect.DeepEqual(foo, fooOut) {
		t.Errorf("expected: %v, saw: %v", foo, fooOut)
	}
}

func TestExtractResourceVersion(t *testing.T) {
	foo := Foo{
		ListMeta: unversioned.ListMeta{
			SelfLink:        "http://foo/bar",
			ResourceVersion: "22",
		},
	}
	
	version, err := extractResourceVersion(&foo)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if version != foo.ResourceVersion {
		t.Errorf("expected: %s, saw: %s", foo.ResourceVersion, version)
	}
}
