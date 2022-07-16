/*
   Copyright © 2021 The CDI Authors

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

package cdi

import (
	"os"
	"path/filepath"

	"github.com/pkg/errors"
)

const (
	// DefaultStaticDir is the default directory for static CDI Specs.
	DefaultStaticDir = "/etc/cdi"
	// DefaultDynamicDir is the default directory for generated CDI Specs
	DefaultDynamicDir = "/var/run/cdi"
)

var (
	// DefaultSpecDirs is the default Spec directory configuration.
	// While altering this variable changes the package defaults,
	// the preferred way of overriding the default directories is
	// to use a WithSpecDirs options. Otherwise the change is only
	// effective if it takes place before creating the Registry or
	// other Cache instances.
	DefaultSpecDirs = []string{DefaultStaticDir, DefaultDynamicDir}
	// ErrStopScan can be returned from a ScanSpecFunc to stop the scan.
	ErrStopScan = errors.New("stop Spec scan")
)

// WithSpecDirs returns an option to override the CDI Spec directories.
func WithSpecDirs(dirs ...string) Option {
	return func(c *Cache) error {
		c.specDirs = make([]string, len(dirs))
		for i, dir := range dirs {
			c.specDirs[i] = filepath.Clean(dir)
		}
		return nil
	}
}

// scanSpecFunc is a function for processing CDI Spec files.
type scanSpecFunc func(string, int, *Spec, error) error

// ScanSpecDirs scans the given directories looking for CDI Spec files,
// which are all files with a '.json' or '.yaml' suffix. For every Spec
// file discovered, ScanSpecDirs loads a Spec from the file then calls
// the scan function passing it the path to the file, the priority (the
// index of the directory in the slice of directories given), the Spec
// itself, and any error encountered while loading the Spec.
//
// Scanning stops once all files have been processed or when the scan
// function returns an error. The result of ScanSpecDirs is the error
// returned by the scan function, if any. The special error ErrStopScan
// can be used to terminate the scan gracefully without ScanSpecDirs
// returning an error. ScanSpecDirs silently skips any subdirectories.
func scanSpecDirs(dirs []string, scanFn scanSpecFunc) error {
	var (
		spec *Spec
		err  error
	)

	for priority, dir := range dirs {
		err = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			// for initial stat failure Walk calls us with nil info
			if info == nil {
				return err
			}
			// first call from Walk is for dir itself, others we skip
			if info.IsDir() {
				if path == dir {
					return nil
				}
				return filepath.SkipDir
			}

			// ignore obviously non-Spec files
			if ext := filepath.Ext(path); ext != ".json" && ext != ".yaml" {
				return nil
			}

			if err != nil {
				return scanFn(path, priority, nil, err)
			}

			spec, err = ReadSpec(path, priority)
			return scanFn(path, priority, spec, err)
		})

		if err != nil && err != ErrStopScan {
			return err
		}
	}

	return nil
}
