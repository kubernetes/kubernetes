package runconfig

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"testing"
)

func TestEntrypointMarshalJSON(t *testing.T) {
	entrypoints := map[*Entrypoint]string{
		nil:                                            "",
		&Entrypoint{}:                                  "null",
		&Entrypoint{[]string{"/bin/sh", "-c", "echo"}}: `["/bin/sh","-c","echo"]`,
	}

	for entrypoint, expected := range entrypoints {
		data, err := entrypoint.MarshalJSON()
		if err != nil {
			t.Fatal(err)
		}
		if string(data) != expected {
			t.Fatalf("Expected %v, got %v", expected, string(data))
		}
	}
}

func TestEntrypointUnmarshalJSON(t *testing.T) {
	parts := map[string][]string{
		"":   {"default", "values"},
		"[]": {},
		`["/bin/sh","-c","echo"]`: {"/bin/sh", "-c", "echo"},
	}
	for json, expectedParts := range parts {
		entrypoint := &Entrypoint{
			[]string{"default", "values"},
		}
		if err := entrypoint.UnmarshalJSON([]byte(json)); err != nil {
			t.Fatal(err)
		}

		actualParts := entrypoint.Slice()
		if len(actualParts) != len(expectedParts) {
			t.Fatalf("Expected %v parts, got %v (%v)", len(expectedParts), len(actualParts), expectedParts)
		}
		for index, part := range actualParts {
			if part != expectedParts[index] {
				t.Fatalf("Expected %v, got %v", expectedParts, actualParts)
				break
			}
		}
	}
}

func TestCommandToString(t *testing.T) {
	commands := map[*Command]string{
		&Command{[]string{""}}:           "",
		&Command{[]string{"one"}}:        "one",
		&Command{[]string{"one", "two"}}: "one two",
	}
	for command, expected := range commands {
		toString := command.ToString()
		if toString != expected {
			t.Fatalf("Expected %v, got %v", expected, toString)
		}
	}
}

func TestCommandMarshalJSON(t *testing.T) {
	commands := map[*Command]string{
		nil:        "",
		&Command{}: "null",
		&Command{[]string{"/bin/sh", "-c", "echo"}}: `["/bin/sh","-c","echo"]`,
	}

	for command, expected := range commands {
		data, err := command.MarshalJSON()
		if err != nil {
			t.Fatal(err)
		}
		if string(data) != expected {
			t.Fatalf("Expected %v, got %v", expected, string(data))
		}
	}
}

func TestCommandUnmarshalJSON(t *testing.T) {
	parts := map[string][]string{
		"":   {"default", "values"},
		"[]": {},
		`["/bin/sh","-c","echo"]`: {"/bin/sh", "-c", "echo"},
	}
	for json, expectedParts := range parts {
		command := &Command{
			[]string{"default", "values"},
		}
		if err := command.UnmarshalJSON([]byte(json)); err != nil {
			t.Fatal(err)
		}

		actualParts := command.Slice()
		if len(actualParts) != len(expectedParts) {
			t.Fatalf("Expected %v parts, got %v (%v)", len(expectedParts), len(actualParts), expectedParts)
		}
		for index, part := range actualParts {
			if part != expectedParts[index] {
				t.Fatalf("Expected %v, got %v", expectedParts, actualParts)
				break
			}
		}
	}
}

func TestDecodeContainerConfig(t *testing.T) {
	fixtures := []struct {
		file       string
		entrypoint *Entrypoint
	}{
		{"fixtures/container_config_1_14.json", NewEntrypoint()},
		{"fixtures/container_config_1_17.json", NewEntrypoint("bash")},
		{"fixtures/container_config_1_19.json", NewEntrypoint("bash")},
	}

	for _, f := range fixtures {
		b, err := ioutil.ReadFile(f.file)
		if err != nil {
			t.Fatal(err)
		}

		c, h, err := DecodeContainerConfig(bytes.NewReader(b))
		if err != nil {
			t.Fatal(fmt.Errorf("Error parsing %s: %v", f, err))
		}

		if c.Image != "ubuntu" {
			t.Fatalf("Expected ubuntu image, found %s\n", c.Image)
		}

		if c.Entrypoint.Len() != f.entrypoint.Len() {
			t.Fatalf("Expected %v, found %v\n", f.entrypoint, c.Entrypoint)
		}

		if h.Memory != 1000 {
			t.Fatalf("Expected memory to be 1000, found %d\n", h.Memory)
		}
	}
}

func TestEntrypointUnmarshalString(t *testing.T) {
	var e *Entrypoint
	echo, err := json.Marshal("echo")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(echo, &e); err != nil {
		t.Fatal(err)
	}

	slice := e.Slice()
	if len(slice) != 1 {
		t.Fatalf("expected 1 element after unmarshal: %q", slice)
	}

	if slice[0] != "echo" {
		t.Fatalf("expected `echo`, got: %q", slice[0])
	}
}

func TestEntrypointUnmarshalSlice(t *testing.T) {
	var e *Entrypoint
	echo, err := json.Marshal([]string{"echo"})
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(echo, &e); err != nil {
		t.Fatal(err)
	}

	slice := e.Slice()
	if len(slice) != 1 {
		t.Fatalf("expected 1 element after unmarshal: %q", slice)
	}

	if slice[0] != "echo" {
		t.Fatalf("expected `echo`, got: %q", slice[0])
	}
}

func TestCommandUnmarshalSlice(t *testing.T) {
	var e *Command
	echo, err := json.Marshal([]string{"echo"})
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(echo, &e); err != nil {
		t.Fatal(err)
	}

	slice := e.Slice()
	if len(slice) != 1 {
		t.Fatalf("expected 1 element after unmarshal: %q", slice)
	}

	if slice[0] != "echo" {
		t.Fatalf("expected `echo`, got: %q", slice[0])
	}
}

func TestCommandUnmarshalString(t *testing.T) {
	var e *Command
	echo, err := json.Marshal("echo")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(echo, &e); err != nil {
		t.Fatal(err)
	}

	slice := e.Slice()
	if len(slice) != 1 {
		t.Fatalf("expected 1 element after unmarshal: %q", slice)
	}

	if slice[0] != "echo" {
		t.Fatalf("expected `echo`, got: %q", slice[0])
	}
}
