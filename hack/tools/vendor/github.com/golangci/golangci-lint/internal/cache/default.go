// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
)

// Default returns the default cache to use.
func Default() (*Cache, error) {
	defaultOnce.Do(initDefaultCache)
	return defaultCache, defaultDirErr
}

var (
	defaultOnce  sync.Once
	defaultCache *Cache
)

// cacheREADME is a message stored in a README in the cache directory.
// Because the cache lives outside the normal Go trees, we leave the
// README as a courtesy to explain where it came from.
const cacheREADME = `This directory holds cached build artifacts from golangci-lint.
`

// initDefaultCache does the work of finding the default cache
// the first time Default is called.
func initDefaultCache() {
	dir := DefaultDir()
	if err := os.MkdirAll(dir, 0744); err != nil {
		log.Fatalf("failed to initialize build cache at %s: %s\n", dir, err)
	}
	if _, err := os.Stat(filepath.Join(dir, "README")); err != nil {
		// Best effort.
		if wErr := os.WriteFile(filepath.Join(dir, "README"), []byte(cacheREADME), 0666); wErr != nil {
			log.Fatalf("Failed to write README file to cache dir %s: %s", dir, err)
		}
	}

	c, err := Open(dir)
	if err != nil {
		log.Fatalf("failed to initialize build cache at %s: %s\n", dir, err)
	}
	defaultCache = c
}

var (
	defaultDirOnce sync.Once
	defaultDir     string
	defaultDirErr  error
)

// DefaultDir returns the effective GOLANGCI_LINT_CACHE setting.
func DefaultDir() string {
	// Save the result of the first call to DefaultDir for later use in
	// initDefaultCache. cmd/go/main.go explicitly sets GOCACHE so that
	// subprocesses will inherit it, but that means initDefaultCache can't
	// otherwise distinguish between an explicit "off" and a UserCacheDir error.

	defaultDirOnce.Do(func() {
		defaultDir = os.Getenv("GOLANGCI_LINT_CACHE")
		if filepath.IsAbs(defaultDir) {
			return
		}
		if defaultDir != "" {
			defaultDirErr = fmt.Errorf("GOLANGCI_LINT_CACHE is not an absolute path")
			return
		}

		// Compute default location.
		dir, err := os.UserCacheDir()
		if err != nil {
			defaultDirErr = fmt.Errorf("GOLANGCI_LINT_CACHE is not defined and %v", err)
			return
		}
		defaultDir = filepath.Join(dir, "golangci-lint")
	})

	return defaultDir
}
