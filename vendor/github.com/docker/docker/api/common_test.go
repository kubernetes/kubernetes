package api

import (
	"io/ioutil"
	"path/filepath"
	"testing"

	"os"
)

// LoadOrCreateTrustKey
func TestLoadOrCreateTrustKeyInvalidKeyFile(t *testing.T) {
	tmpKeyFolderPath, err := ioutil.TempDir("", "api-trustkey-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpKeyFolderPath)

	tmpKeyFile, err := ioutil.TempFile(tmpKeyFolderPath, "keyfile")
	if err != nil {
		t.Fatal(err)
	}

	if _, err := LoadOrCreateTrustKey(tmpKeyFile.Name()); err == nil {
		t.Fatal("expected an error, got nothing.")
	}

}

func TestLoadOrCreateTrustKeyCreateKey(t *testing.T) {
	tmpKeyFolderPath, err := ioutil.TempDir("", "api-trustkey-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpKeyFolderPath)

	// Without the need to create the folder hierarchy
	tmpKeyFile := filepath.Join(tmpKeyFolderPath, "keyfile")

	if key, err := LoadOrCreateTrustKey(tmpKeyFile); err != nil || key == nil {
		t.Fatalf("expected a new key file, got : %v and %v", err, key)
	}

	if _, err := os.Stat(tmpKeyFile); err != nil {
		t.Fatalf("Expected to find a file %s, got %v", tmpKeyFile, err)
	}

	// With the need to create the folder hierarchy as tmpKeyFie is in a path
	// where some folders do not exist.
	tmpKeyFile = filepath.Join(tmpKeyFolderPath, "folder/hierarchy/keyfile")

	if key, err := LoadOrCreateTrustKey(tmpKeyFile); err != nil || key == nil {
		t.Fatalf("expected a new key file, got : %v and %v", err, key)
	}

	if _, err := os.Stat(tmpKeyFile); err != nil {
		t.Fatalf("Expected to find a file %s, got %v", tmpKeyFile, err)
	}

	// With no path at all
	defer os.Remove("keyfile")
	if key, err := LoadOrCreateTrustKey("keyfile"); err != nil || key == nil {
		t.Fatalf("expected a new key file, got : %v and %v", err, key)
	}

	if _, err := os.Stat("keyfile"); err != nil {
		t.Fatalf("Expected to find a file keyfile, got %v", err)
	}
}

func TestLoadOrCreateTrustKeyLoadValidKey(t *testing.T) {
	tmpKeyFile := filepath.Join("fixtures", "keyfile")

	if key, err := LoadOrCreateTrustKey(tmpKeyFile); err != nil || key == nil {
		t.Fatalf("expected a key file, got : %v and %v", err, key)
	}
}
