/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package manifest provides shared utilities for loading admission configurations
// from static manifest files.
package manifest

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
)

// splitYAMLDocuments splits a multi-document YAML byte slice into individual documents.
// Empty documents are skipped.
func splitYAMLDocuments(data []byte) ([][]byte, error) {
	reader := utilyaml.NewYAMLReader(bufio.NewReader(bytes.NewReader(data)))
	var docs [][]byte
	for {
		doc, err := reader.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}
		doc = bytes.TrimSpace(doc)
		if len(doc) == 0 {
			continue
		}
		docs = append(docs, doc)
	}
	return docs, nil
}

// FileDoc holds a decoded YAML document and the file it came from.
type FileDoc struct {
	FilePath string
	Doc      []byte
}

// LoadFiles reads all YAML/JSON files from dir, splits multi-document YAML,
// and returns individual documents with their source file paths plus a
// sha256-prefixed hash of the file contents for change detection.
// Files are processed in alphabetical order for deterministic behavior.
func LoadFiles(dir string) ([]FileDoc, string, error) {
	if len(dir) == 0 {
		return nil, "", fmt.Errorf("manifest directory path is empty")
	}

	// os.ReadDir returns entries sorted by filename.
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, "", fmt.Errorf("failed to read manifest directory %q: %w", dir, err)
	}

	var fileDocs []FileDoc
	h := sha256.New()
	hasData := false

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		ext := strings.ToLower(filepath.Ext(name))
		if ext != ".yaml" && ext != ".yml" && ext != ".json" {
			continue
		}

		filePath := filepath.Join(dir, name)
		data, err := os.ReadFile(filePath)
		if err != nil {
			return nil, "", fmt.Errorf("failed to read file %q: %w", filePath, err)
		}
		if len(data) == 0 {
			continue
		}

		h.Write(data)
		h.Write([]byte{0})
		hasData = true

		docs, err := splitYAMLDocuments(data)
		if err != nil {
			return nil, "", fmt.Errorf("failed to split YAML documents in file %q: %w", filePath, err)
		}
		for _, doc := range docs {
			fileDocs = append(fileDocs, FileDoc{FilePath: filePath, Doc: doc})
		}
	}

	var hash string
	if hasData {
		hash = fmt.Sprintf("sha256:%x", h.Sum(nil))
	}

	return fileDocs, hash, nil
}
