// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package loadertest holds a fake for the Loader interface.
package loadertest

import (
	"log"
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/loader"
	"sigs.k8s.io/kustomize/pkg/types"
	"sigs.k8s.io/kustomize/pkg/validators"
)

// FakeLoader encapsulates the delegate Loader and the fake file system.
type FakeLoader struct {
	fs       fs.FileSystem
	delegate ifc.Loader
}

// NewFakeLoader returns a Loader that uses a fake filesystem.
// The loader will be restricted to root only.
// The initialDir argument should be an absolute file path.
func NewFakeLoader(initialDir string) FakeLoader {
	return NewFakeLoaderWithRestrictor(
		loader.RestrictionRootOnly, initialDir)
}

// NewFakeLoaderWithRestrictor returns a Loader that
// uses a fake filesystem.
// The initialDir argument should be an absolute file path.
func NewFakeLoaderWithRestrictor(
	lr loader.LoadRestrictorFunc, initialDir string) FakeLoader {
	// Create fake filesystem and inject it into initial Loader.
	fSys := fs.MakeFakeFS()
	fSys.Mkdir(initialDir)
	ldr, err := loader.NewLoader(
		lr, validators.MakeFakeValidator(), initialDir, fSys)
	if err != nil {
		log.Fatalf("Unable to make loader: %v", err)
	}
	return FakeLoader{fs: fSys, delegate: ldr}
}

// AddFile adds a fake file to the file system.
func (f FakeLoader) AddFile(fullFilePath string, content []byte) error {
	return f.fs.WriteFile(fullFilePath, content)
}

// AddDirectory adds a fake directory to the file system.
func (f FakeLoader) AddDirectory(fullDirPath string) error {
	return f.fs.Mkdir(fullDirPath)
}

// Root delegates.
func (f FakeLoader) Root() string {
	return f.delegate.Root()
}

// New creates a new loader from a new root.
func (f FakeLoader) New(newRoot string) (ifc.Loader, error) {
	l, err := f.delegate.New(newRoot)
	if err != nil {
		return nil, err
	}
	return FakeLoader{fs: f.fs, delegate: l}, nil
}

// Load delegates.
func (f FakeLoader) Load(location string) ([]byte, error) {
	return f.delegate.Load(location)
}

// Cleanup delegates.
func (f FakeLoader) Cleanup() error {
	return f.delegate.Cleanup()
}

// Validator delegates.
func (f FakeLoader) Validator() ifc.Validator {
	return f.delegate.Validator()
}

// LoadKvPairs delegates.
func (f FakeLoader) LoadKvPairs(args types.GeneratorArgs) ([]types.Pair, error) {
	return f.delegate.LoadKvPairs(args)
}
