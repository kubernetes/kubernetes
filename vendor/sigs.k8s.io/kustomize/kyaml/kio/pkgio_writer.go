// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package kio

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/filesys"
	"sigs.k8s.io/kustomize/kyaml/kio/kioutil"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// LocalPackageWriter writes ResourceNodes to a filesystem
type LocalPackageWriter struct {
	Kind string `yaml:"kind,omitempty"`

	// PackagePath is the path to the package directory.
	PackagePath string `yaml:"path,omitempty"`

	// KeepReaderAnnotations if set will retain the annotations set by LocalPackageReader
	KeepReaderAnnotations bool `yaml:"keepReaderAnnotations,omitempty"`

	// ClearAnnotations will clear annotations before writing the resources
	ClearAnnotations []string `yaml:"clearAnnotations,omitempty"`

	// FileSystem can be used to mock the disk file system.
	FileSystem filesys.FileSystemOrOnDisk
}

var _ Writer = LocalPackageWriter{}

func (r LocalPackageWriter) Write(nodes []*yaml.RNode) error {
	// set the path and index annotations if they are missing
	if err := kioutil.DefaultPathAndIndexAnnotation("", nodes); err != nil {
		return err
	}

	if !r.FileSystem.Exists(r.PackagePath) {
		return errors.WrapPrefixf(os.ErrNotExist, "could not write to %q", r.PackagePath)
	}
	if !r.FileSystem.IsDir(r.PackagePath) {
		// if the user specified input isn't a directory, the package is the directory of the
		// target
		r.PackagePath = filepath.Dir(r.PackagePath)
	}

	// setup indexes for writing Resources back to files
	if err := r.errorIfMissingRequiredAnnotation(nodes); err != nil {
		return err
	}
	outputFiles, err := r.indexByFilePath(nodes)
	if err != nil {
		return err
	}
	for k := range outputFiles {
		if err = kioutil.SortNodes(outputFiles[k]); err != nil {
			return errors.Wrap(err)
		}
	}

	if !r.KeepReaderAnnotations {
		r.ClearAnnotations = append(r.ClearAnnotations, kioutil.PathAnnotation)
		r.ClearAnnotations = append(r.ClearAnnotations, kioutil.LegacyPathAnnotation)
	}

	// validate outputs before writing any
	for path := range outputFiles {
		outputPath := filepath.Join(r.PackagePath, path)
		if r.FileSystem.IsDir(outputPath) {
			return fmt.Errorf("config.kubernetes.io/path cannot be a directory: %s", path)
		}

		err = r.FileSystem.MkdirAll(filepath.Dir(outputPath))
		if err != nil {
			return errors.Wrap(err)
		}
	}

	// write files
	buf := bytes.NewBuffer(nil)
	for path := range outputFiles {
		outputPath := filepath.Join(r.PackagePath, path)
		err = r.FileSystem.MkdirAll(filepath.Dir(filepath.Join(r.PackagePath, path)))
		if err != nil {
			return errors.Wrap(err)
		}

		buf.Reset()
		w := ByteWriter{
			Writer:                buf,
			KeepReaderAnnotations: r.KeepReaderAnnotations,
			ClearAnnotations:      r.ClearAnnotations,
		}
		if err = w.Write(outputFiles[path]); err != nil {
			return errors.Wrap(err)
		}

		if err := r.FileSystem.WriteFile(outputPath, buf.Bytes()); err != nil {
			return errors.Wrap(err)
		}
	}

	return nil
}

func (r LocalPackageWriter) errorIfMissingRequiredAnnotation(nodes []*yaml.RNode) error {
	for i := range nodes {
		for _, s := range requiredResourcePackageAnnotations {
			key, err := nodes[i].Pipe(yaml.GetAnnotation(s))
			if err != nil {
				return errors.Wrap(err)
			}
			if key == nil || key.YNode() == nil || key.YNode().Value == "" {
				return errors.Errorf(
					"resources must be annotated with %s to be written to files", s)
			}
		}
	}
	return nil
}

func (r LocalPackageWriter) indexByFilePath(nodes []*yaml.RNode) (map[string][]*yaml.RNode, error) {
	outputFiles := map[string][]*yaml.RNode{}
	for i := range nodes {
		// parse the file write path
		node := nodes[i]
		value, err := node.Pipe(yaml.GetAnnotation(kioutil.PathAnnotation))
		if err != nil {
			// this should never happen if errorIfMissingRequiredAnnotation was run
			return nil, errors.Wrap(err)
		}
		path := value.YNode().Value
		outputFiles[path] = append(outputFiles[path], node)

		if filepath.IsAbs(path) {
			return nil, errors.Errorf("package paths may not be absolute paths")
		}
		if strings.Contains(filepath.Clean(path), "..") {
			return nil, fmt.Errorf("resource must be written under package %s: %s",
				r.PackagePath, filepath.Clean(path))
		}
	}
	return outputFiles, nil
}
