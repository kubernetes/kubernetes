/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package main

import (
	"bytes"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	flag "github.com/spf13/pflag"
)

var (
	verify  = flag.Bool("verify", false, "Exit with status 1 if files would have needed changes but do not change.")
	rootDir = flag.String("root-dir", "", "Root directory containing documents to be processed.")

	ErrChangesNeeded = errors.New("mungedocs: changes required")

	// TODO: allow selection from command line. (e.g., just check links in the examples directory.)
	mungesToMake = munges{
		munger(updateTOC),
		munger(checkLinks),
	}
)

// Munger processes a document, returning an updated document xor an error.
// Munger is NOT allowed to mutate 'before', if changes are needed it must copy
// data into a new byte array.
type munger func(filePath string, before []byte) (after []byte, err error)

type munges []munger

type fileProcessor struct {
	// Which munge functions should we call?
	munges munges

	// Are we allowed to make changes?
	verifyOnly bool
}

// Either change a file or verify that it needs no changes (according to modify argument)
func (f fileProcessor) visit(path string, i os.FileInfo, e error) error {
	if !strings.HasSuffix(path, ".md") {
		return nil
	}

	fileBytes, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	modificationsMade := false
	for _, munge := range f.munges {
		after, err := munge(path, fileBytes)
		if err != nil {
			return err
		}
		if !modificationsMade {
			if !bytes.Equal(after, fileBytes) {
				modificationsMade = true
				if f.verifyOnly {
					// We're not allowed to make changes.
					return ErrChangesNeeded
				}
			}
		}
		fileBytes = after
	}

	// Write out new file with any changes.
	if modificationsMade {
		ioutil.WriteFile(path, fileBytes, 0644)
	}

	return nil
}

func main() {
	flag.Parse()

	if *rootDir == "" {
		fmt.Fprintf(os.Stderr, "usage: %s [--verify] --root-dir <docs root>\n", flag.Arg(0))
		os.Exit(1)
	}

	fp := fileProcessor{
		munges:     mungesToMake,
		verifyOnly: *verify,
	}

	// For each markdown file under source docs root, process the doc.
	// If any error occurs, will exit with failure.
	// If verify is true, then status is 0 for no changes needed, 1 for changes needed
	// and >1 for an error during processing.
	// If verify is false, then status is 0 if changes successfully made or no changes needed,
	// 1 if changes were needed but require human intervention, and >1 for an unexpected
	// error during processing.
	err := filepath.Walk(*rootDir, fp.visit)
	if err != nil {
		if err == ErrChangesNeeded {
			if *verify {
				fmt.Fprintf(os.Stderr,
					"Some changes needed but not made due to --verify=true\n")
			} else {
				fmt.Fprintf(os.Stderr,
					"Some changes needed but human intervention is required\n")
			}
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "filepath.Walk() returned %v\n", err)
		os.Exit(2)
	}
}
