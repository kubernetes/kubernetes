package stressClient

import (
	"testing"
)

func TestNewTracer(t *testing.T) {
	tagValue := "foo_tag_value"
	tracer := NewTracer(map[string]string{"foo_tag_key": tagValue})
	got := tracer.Tags["foo_tag_key"]
	if got != tagValue {
		t.Errorf("expected: %v\ngot: %v", tagValue, got)
	}
	tracer.Add(1)
	tracer.Done()
	tracer.Wait()
}
