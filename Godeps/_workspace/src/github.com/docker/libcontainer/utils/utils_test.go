package utils

import "testing"

func TestGenerateName(t *testing.T) {
	name, err := GenerateRandomName("veth", 5)
	if err != nil {
		t.Fatal(err)
	}

	expected := 5 + len("veth")
	if len(name) != 5+len("veth") {
		t.Fatalf("expected name to be %d chars but received %d", expected, len(name))
	}
}
