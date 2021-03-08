// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package kio

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

// Setup creates directories and files for testing
type Setup struct {
	// root is the tmp directory
	Root string
}

// setupDirectories creates directories for reading test configuration from
func SetupDirectories(t *testing.T, dirs ...string) Setup {
	d, err := ioutil.TempDir("", "kyaml-test")
	if !assert.NoError(t, err) {
		assert.FailNow(t, err.Error())
	}
	err = os.Chdir(d)
	if !assert.NoError(t, err) {
		assert.FailNow(t, err.Error())
	}
	for _, s := range dirs {
		err = os.MkdirAll(s, 0700)
		if !assert.NoError(t, err) {
			assert.FailNow(t, err.Error())
		}
	}
	return Setup{Root: d}
}

// writeFile writes a file under the test directory
func (s Setup) WriteFile(t *testing.T, path string, value []byte) {
	err := os.MkdirAll(filepath.Dir(filepath.Join(s.Root, path)), 0700)
	if !assert.NoError(t, err) {
		assert.FailNow(t, err.Error())
	}
	err = ioutil.WriteFile(filepath.Join(s.Root, path), value, 0600)
	if !assert.NoError(t, err) {
		assert.FailNow(t, err.Error())
	}
}

// clean deletes the test config
func (s Setup) Clean() {
	os.RemoveAll(s.Root)
}
