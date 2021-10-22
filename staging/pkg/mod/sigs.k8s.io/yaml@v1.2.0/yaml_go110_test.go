// +build go1.10

package yaml

import (
	"fmt"
	"testing"
)

func TestUnmarshalWithTags(t *testing.T) {
	type WithTaggedField struct {
		Field string `json:"field"`
	}

	t.Run("Known tagged field", func(t *testing.T) {
		y := []byte(`field: "hello"`)
		v := WithTaggedField{}
		if err := Unmarshal(y, &v, DisallowUnknownFields); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if v.Field != "hello" {
			t.Errorf("v.Field=%v, want 'hello'", v.Field)
		}

	})
	t.Run("With unknown tagged field", func(t *testing.T) {
		y := []byte(`unknown: "hello"`)
		v := WithTaggedField{}
		err := Unmarshal(y, &v, DisallowUnknownFields)
		if err == nil {
			t.Errorf("want error because of unknown field, got <nil>: v=%#v", v)
		}
	})

}

func exampleUnknown() {
	type WithTaggedField struct {
		Field string `json:"field"`
	}
	y := []byte(`unknown: "hello"`)
	v := WithTaggedField{}
	fmt.Printf("%v\n", Unmarshal(y, &v, DisallowUnknownFields))
	// Ouptut:
	// unmarshaling JSON: while decoding JSON: json: unknown field "unknown"
}
