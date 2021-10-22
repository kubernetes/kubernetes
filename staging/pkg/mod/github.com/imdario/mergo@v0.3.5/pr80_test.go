package mergo

import (
	"testing"
)

type mapInterface map[string]interface{}

func TestMergeMapsEmptyString(t *testing.T) {
	a := mapInterface{"s": ""}
	b := mapInterface{"s": "foo"}
	if err := Merge(&a, b); err != nil {
		t.Fatal(err)
	}
	if a["s"] != "foo" {
		t.Fatalf("b not merged in properly: a.s.Value(%s) != expected(%s)", a["s"], "foo")
	}
}
