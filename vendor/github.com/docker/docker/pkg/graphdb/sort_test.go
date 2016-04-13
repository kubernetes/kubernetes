package graphdb

import (
	"testing"
)

func TestSort(t *testing.T) {
	paths := []string{
		"/",
		"/myreallylongname",
		"/app/db",
	}

	sortByDepth(paths)

	if len(paths) != 3 {
		t.Fatalf("Expected 3 parts got %d", len(paths))
	}

	if paths[0] != "/app/db" {
		t.Fatalf("Expected /app/db got %s", paths[0])
	}
	if paths[1] != "/myreallylongname" {
		t.Fatalf("Expected /myreallylongname got %s", paths[1])
	}
	if paths[2] != "/" {
		t.Fatalf("Expected / got %s", paths[2])
	}
}
