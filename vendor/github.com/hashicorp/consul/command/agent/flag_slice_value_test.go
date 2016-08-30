package agent

import (
	"flag"
	"reflect"
	"testing"
)

func TestAppendSliceValue_implements(t *testing.T) {
	var raw interface{}
	raw = new(AppendSliceValue)
	if _, ok := raw.(flag.Value); !ok {
		t.Fatalf("AppendSliceValue should be a Value")
	}
}

func TestAppendSliceValueSet(t *testing.T) {
	sv := new(AppendSliceValue)
	err := sv.Set("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	err = sv.Set("bar")
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	expected := []string{"foo", "bar"}
	if !reflect.DeepEqual([]string(*sv), expected) {
		t.Fatalf("Bad: %#v", sv)
	}
}
