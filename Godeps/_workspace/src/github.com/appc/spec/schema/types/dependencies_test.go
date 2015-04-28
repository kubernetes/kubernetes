package types

import "testing"

func TestEmptyHash(t *testing.T) {
	dj := `{"app": "example.com/reduce-worker-base"}`

	var d Dependency

	err := d.UnmarshalJSON([]byte(dj))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Marshal to verify that marshalling works without validation errors
	buf, err := d.MarshalJSON()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Unmarshal to verify that the generated json will not create wrong empty hash
	err = d.UnmarshalJSON(buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
