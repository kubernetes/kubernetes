package mapstructure

import "testing"

// GH-1
func TestDecode_NilValue(t *testing.T) {
	input := map[string]interface{}{
		"vfoo":   nil,
		"vother": nil,
	}

	var result Map
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("should not error: %s", err)
	}

	if result.Vfoo != "" {
		t.Fatalf("value should be default: %s", result.Vfoo)
	}

	if result.Vother != nil {
		t.Fatalf("Vother should be nil: %s", result.Vother)
	}
}

// GH-10
func TestDecode_mapInterfaceInterface(t *testing.T) {
	input := map[interface{}]interface{}{
		"vfoo":   nil,
		"vother": nil,
	}

	var result Map
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("should not error: %s", err)
	}

	if result.Vfoo != "" {
		t.Fatalf("value should be default: %s", result.Vfoo)
	}

	if result.Vother != nil {
		t.Fatalf("Vother should be nil: %s", result.Vother)
	}
}
