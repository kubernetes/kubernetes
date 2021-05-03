/*
Copyright 2021 The Kubernetes Authors.

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

package startupcrd

import (
	"bytes"
	"embed"
	"io"
	"io/fs"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	yamlserializer "k8s.io/apimachinery/pkg/runtime/serializer/yaml"
	"k8s.io/apimachinery/pkg/util/yaml"
	"os"
	"path/filepath"
	"strings"
)

//go:embed configs/*yaml
var inbuiltCRDs embed.FS

// Reader defines the methods an implementation should have
type Reader interface {
	Read() ([]*unstructured.Unstructured, error)
}

// Readers encapsulates multiple readers
type Readers []Reader

// Read returns the objects read from the underlying Readers
func (readers Readers) Read() ([]*unstructured.Unstructured, error) {
	var objs []*unstructured.Unstructured
	for _, r := range readers {
		obj, err := r.Read()
		if err != nil {
			return nil, err
		}
		objs = append(objs, obj...)
	}

	return objs, nil
}

// EmbeddedFSReader implements Reader for an embedded filesystem
type EmbeddedFSReader struct {
	FS embed.FS
}

// Read returns objects read from an embedded filesystem
func (r EmbeddedFSReader) Read() ([]*unstructured.Unstructured, error) {
	var objs []*unstructured.Unstructured

	// walk the embedded filesystem
	if err := fs.WalkDir(r.FS, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if !d.IsDir() && strings.HasSuffix(path, ".yaml") {
			f, err := r.FS.Open(path)
			if err != nil {
				return err
			}

			defer f.Close()

			readObjs, err := ReadFromIOReader(f)
			if err != nil {
				return err
			}

			objs = append(objs, readObjs...)
		}

		return nil
	}); err != nil {
		return nil, err
	}

	return objs, nil
}

// GetInbuiltEmbeddedFSReader returns an EmbeddedFSReader which stores
// an embedded filesystem pointing to the inbuilt startups CRDs
func GetInbuiltEmbeddedFSReader() EmbeddedFSReader {
	return EmbeddedFSReader{FS: inbuiltCRDs}
}

// FSReader implements Reader for an actual filesystem
type FSReader struct {
	ManifestDirectory string
}

// Read reads Kubernetes objects from an actual filesystem
func (r FSReader) Read() ([]*unstructured.Unstructured, error) {
	var objs []*unstructured.Unstructured

	if err := filepath.Walk(r.ManifestDirectory, func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if !info.IsDir() && strings.HasSuffix(path, ".yaml") {
			f, err := os.Open(path)
			if err != nil {
				return err
			}

			defer f.Close()

			readObjs, err := ReadFromIOReader(f)
			if err != nil {
				return err
			}

			objs = append(objs, readObjs...)
		}

		return nil
	}); err != nil {
		return nil, err
	}

	return objs, nil
}

// ReadFromIOReader takes an io.Reader, reads the Kubernetes manifests
// and returns the unstructured objects
func ReadFromIOReader(r io.Reader) ([]*unstructured.Unstructured, error) {
	var objs []*unstructured.Unstructured

	decoder := yaml.NewYAMLOrJSONDecoder(r, 4098)
	serializer := yamlserializer.NewDecodingSerializer(unstructured.UnstructuredJSONScheme)

	for {
		ext := &runtime.RawExtension{}
		if err := decoder.Decode(ext); err != nil {
			if err == io.EOF {
				break
			}

			return nil, err
		}

		ext.Raw = bytes.TrimSpace(ext.Raw)
		if len(ext.Raw) == 0 || bytes.Equal(ext.Raw, []byte("null")) {
			continue
		}

		obj := &unstructured.Unstructured{}

		if _, _, err := serializer.Decode(ext.Raw, nil, obj); err != nil {
			return nil, err
		}

		objs = append(objs, obj)
	}

	return objs, nil
}
