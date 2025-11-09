package json

import (
	"bytes"
	"testing"
)

type unknownCapture struct {
	Known         string               `json:"known"`
	UnknownFields map[string]any       `json:"-"`
	Nested        nestedUnknownCapture `json:"nested,omitempty"`
}

type nestedUnknownCapture struct {
	Value         string         `json:"value"`
	UnknownFields map[string]any `json:"-"`
}

type EmbeddedUnknownHolder struct {
	UnknownFields map[string]any `json:"-"`
}

func TestUnknownFieldsCapture(t *testing.T) {
	data := []byte(`{"known":"ok","mystery":42,"nested":{"value":"x","extra":{"foo":"bar"}}}`)
	var out unknownCapture
	if err := Unmarshal(data, &out); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if out.UnknownFields == nil {
		t.Fatalf("expected UnknownFields to be populated")
	}
	if got := out.UnknownFields["mystery"]; got != float64(42) {
		t.Fatalf("expected mystery=42, got %#v", got)
	}

	if out.Nested.UnknownFields == nil {
		t.Fatalf("expected nested UnknownFields to be populated")
	}
	nestedExtra, ok := out.Nested.UnknownFields["extra"].(map[string]any)
	if !ok || nestedExtra["foo"] != "bar" {
		t.Fatalf("expected nested extra map, got %#v", out.Nested.UnknownFields["extra"])
	}
}

func TestUnknownFieldsResetBetweenDecodes(t *testing.T) {
	var out unknownCapture
	first := []byte(`{"known":"one","first":1}`)
	second := []byte(`{"known":"two"}`)

	if err := Unmarshal(first, &out); err != nil {
		t.Fatalf("unexpected error on first decode: %v", err)
	}
	if len(out.UnknownFields) != 1 {
		t.Fatalf("expected 1 unknown field after first decode, got %d", len(out.UnknownFields))
	}

	if err := Unmarshal(second, &out); err != nil {
		t.Fatalf("unexpected error on second decode: %v", err)
	}
	if out.UnknownFields != nil && len(out.UnknownFields) != 0 {
		t.Fatalf("expected unknown fields to reset, got %#v", out.UnknownFields)
	}
}

func TestUnknownFieldsAllocatesEmbeddedPointers(t *testing.T) {
	type outer struct {
		*EmbeddedUnknownHolder
	}

	var out outer
	data := []byte(`{"extra":"value"}`)
	if err := Unmarshal(data, &out); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.EmbeddedUnknownHolder == nil {
		t.Fatalf("expected embedded pointer to be allocated")
	}
	if got := out.UnknownFields["extra"]; got != "value" {
		t.Fatalf("expected extra=\"value\", got %#v", got)
	}
}

func TestUnknownFieldsWithDisallowUnknown(t *testing.T) {
	var out unknownCapture
	data := []byte(`{"known":"ok","mystery":42}`)

	dec := NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&out); err == nil {
		t.Fatalf("expected error when disallowing unknown fields")
	}
	if out.UnknownFields != nil {
		t.Fatalf("expected UnknownFields to stay nil when decode failed, got %#v", out.UnknownFields)
	}
}
