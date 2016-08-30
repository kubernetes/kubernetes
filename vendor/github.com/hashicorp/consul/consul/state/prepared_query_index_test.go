package state

import (
	"testing"

	"github.com/hashicorp/consul/consul/structs"
)

// Indexer is a global indexer to use for tests since there is no state.
var index PreparedQueryIndex

func TestPreparedQueryIndex_FromObject(t *testing.T) {
	// We shouldn't index an object we don't understand.
	if ok, _, err := index.FromObject(42); ok || err == nil {
		t.Fatalf("bad: ok=%v err=%v", ok, err)
	}

	// We shouldn't index a non-template query.
	wrapped := &queryWrapper{&structs.PreparedQuery{}, nil}
	if ok, _, err := index.FromObject(wrapped); ok || err != nil {
		t.Fatalf("bad: ok=%v err=%v", ok, err)
	}

	// Create a template with an empty name.
	query := &structs.PreparedQuery{
		Template: structs.QueryTemplateOptions{
			Type: structs.QueryTemplateTypeNamePrefixMatch,
		},
	}
	ok, key, err := index.FromObject(&queryWrapper{query, nil})
	if !ok || err != nil {
		t.Fatalf("bad: ok=%v err=%v", ok, err)
	}
	if string(key) != "\x00" {
		t.Fatalf("bad: %#v", key)
	}

	// Set the name and try again.
	query.Name = "hello"
	ok, key, err = index.FromObject(&queryWrapper{query, nil})
	if !ok || err != nil {
		t.Fatalf("bad: ok=%v err=%v", ok, err)
	}
	if string(key) != "\x00hello" {
		t.Fatalf("bad: %#v", key)
	}

	// Make sure the index isn't case-sensitive.
	query.Name = "HELLO"
	ok, key, err = index.FromObject(&queryWrapper{query, nil})
	if !ok || err != nil {
		t.Fatalf("bad: ok=%v err=%v", ok, err)
	}
	if string(key) != "\x00hello" {
		t.Fatalf("bad: %#v", key)
	}
}

func TestPreparedQueryIndex_FromArgs(t *testing.T) {
	// Only accept a single string arg.
	if _, err := index.FromArgs(42); err == nil {
		t.Fatalf("should be an error")
	}
	if _, err := index.FromArgs("hello", "world"); err == nil {
		t.Fatalf("should be an error")
	}

	// Try an empty string.
	if key, err := index.FromArgs(""); err != nil || string(key) != "\x00" {
		t.Fatalf("bad: key=%#v err=%v", key, err)
	}

	// Try a non-empty string.
	if key, err := index.FromArgs("hello"); err != nil ||
		string(key) != "\x00hello" {
		t.Fatalf("bad: key=%#v err=%v", key, err)
	}

	// Make sure index is not case-sensitive.
	if key, err := index.FromArgs("HELLO"); err != nil ||
		string(key) != "\x00hello" {
		t.Fatalf("bad: key=%#v err=%v", key, err)
	}
}

func TestPreparedQueryIndex_PrefixFromArgs(t *testing.T) {
	// Only accept a single string arg.
	if _, err := index.PrefixFromArgs(42); err == nil {
		t.Fatalf("should be an error")
	}
	if _, err := index.PrefixFromArgs("hello", "world"); err == nil {
		t.Fatalf("should be an error")
	}

	// Try an empty string.
	if key, err := index.PrefixFromArgs(""); err != nil || string(key) != "\x00" {
		t.Fatalf("bad: key=%#v err=%v", key, err)
	}

	// Try a non-empty string.
	if key, err := index.PrefixFromArgs("hello"); err != nil ||
		string(key) != "\x00hello" {
		t.Fatalf("bad: key=%#v err=%v", key, err)
	}

	// Make sure index is not case-sensitive.
	if key, err := index.PrefixFromArgs("HELLO"); err != nil ||
		string(key) != "\x00hello" {
		t.Fatalf("bad: key=%#v err=%v", key, err)
	}
}
