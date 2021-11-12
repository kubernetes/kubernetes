// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package kio

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// Setup creates directories and files for testing
type Setup struct {
	// root is the tmp directory
	Root string
}

// setupDirectories creates directories for reading test configuration from
func SetupDirectories(t *testing.T, dirs ...string) Setup {
	d, err := ioutil.TempDir("", "kyaml-test")
	require.NoError(t, err)
	err = os.Chdir(d)
	require.NoError(t, err)
	for _, s := range dirs {
		err = os.MkdirAll(s, 0700)
		require.NoError(t, err)
	}
	return Setup{Root: d}
}

// writeFile writes a file under the test directory
func (s Setup) WriteFile(t *testing.T, path string, value []byte) {
	err := os.MkdirAll(filepath.Dir(filepath.Join(s.Root, path)), 0700)
	require.NoError(t, err)
	err = ioutil.WriteFile(filepath.Join(s.Root, path), value, 0600)
	require.NoError(t, err)
}

// clean deletes the test config
func (s Setup) Clean() {
	os.RemoveAll(s.Root)
}
