package golist

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
)

// Package is `go list --json` output structure.
type Package struct {
	Dir        string   // directory containing package sources
	ImportPath string   // import path of package in dir
	GoFiles    []string // .go source files (excluding CgoFiles, TestGoFiles, XTestGoFiles)
}

// JSON runs `go list --json` for the specified pkgName and returns the parsed JSON.
func JSON(pkgPath string) (*Package, error) {
	out, err := exec.Command("go", "list", "--json", pkgPath).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("go list error (%v): %s", err, out)
	}

	var pkg Package
	if err := json.NewDecoder(bytes.NewReader(out)).Decode(&pkg); err != io.EOF && err != nil {
		return nil, err
	}
	return &pkg, nil
}
