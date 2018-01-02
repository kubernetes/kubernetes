package stressClient

import (
	"testing"
)

func TestNewResponse(t *testing.T) {
	pt := NewBlankTestPoint()
	tr := NewTracer(map[string]string{})
	r := NewResponse(pt, tr)
	expected := "another_tag_value"
	test := r.AddTags(map[string]string{"another_tag": "another_tag_value"})
	got := test.Tags()["another_tag"]
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
}
