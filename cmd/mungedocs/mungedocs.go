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
	"path"
	"path/filepath"
	"strings"

	flag "github.com/spf13/pflag"
)

var (
	verify   = flag.Bool("verify", false, "Exit with status 1 if files would have needed changes but do not change.")
	rootDir  = flag.String("root-dir", "", "Root directory containing documents to be processed.")
	repoRoot = flag.String("repo-root", "..", `Appended to --root-dir to get the repository root.
It's done this way so that generally you just have to set --root-dir.
Examples:
 * --root-dir=docs/ --repo-root=.. means the repository root is ./
 * --root-dir=/usr/local/long/path/repo/docs/ --repo-root=.. means the repository root is /usr/local/long/path/repo/
 * --root-dir=/usr/local/long/path/repo/docs/admin --repo-root=../.. means the repository root is /usr/local/long/path/repo/`)
	skipMunges = flag.String("skip-munges", "", "Comma-separated list of munges to *not* run. Available munges are: "+availableMungeList)

	ErrChangesNeeded = errors.New("mungedocs: changes required")

	// All of the munge operations to perform.
	// TODO: allow selection from command line. (e.g., just check links in the examples directory.)
	allMunges = []munge{
		{"table-of-contents", updateTOC},
		{"unversioned-warning", updateUnversionedWarning},
		{"check-links", checkLinks},
		{"blank-lines-surround-preformatted", checkPreformatted},
		{"header-lines", checkHeaderLines},
		{"analytics", checkAnalytics},
		{"kubectl-dash-f", checkKubectlFileTargets},
	}
	availableMungeList = func() string {
		names := []string{}
		for _, m := range allMunges {
			names = append(names, m.name)
		}
		return strings.Join(names, ",")
	}()
)

// a munge processes a document, returning an updated document xor an error.
// The fn is NOT allowed to mutate 'before', if changes are needed it must copy
// data into a new byte array and return that.
type munge struct {
	name string
	fn   func(filePath string, before []byte) (after []byte, err error)
}

type fileProcessor struct {
	// Which munge functions should we call?
	munges []munge

	// Are we allowed to make changes?
	verifyOnly bool
}

// Either change a file or verify that it needs no changes (according to modify argument)
func (f fileProcessor) visit(path string) error {
	if !strings.HasSuffix(path, ".md") {
		return nil
	}

	fileBytes, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	modificationsMade := false
	errFound := false
	filePrinted := false
	for _, munge := range f.munges {
		after, err := munge.fn(path, fileBytes)
		if err != nil || !bytes.Equal(after, fileBytes) {
			if !filePrinted {
				fmt.Printf("%s\n----\n", path)
				filePrinted = true
			}
			fmt.Printf("%s:\n", munge.name)
			if err != nil {
				fmt.Println(err)
				errFound = true
			} else {
				fmt.Println("contents were modified")
				modificationsMade = true
			}
			fmt.Println("")
		}
		fileBytes = after
	}

	// Write out new file with any changes.
	if modificationsMade {
		if f.verifyOnly {
			// We're not allowed to make changes.
			return ErrChangesNeeded
		}
		ioutil.WriteFile(path, fileBytes, 0644)
	}
	if errFound {
		return ErrChangesNeeded
	}

	return nil
}

func newWalkFunc(fp *fileProcessor, changesNeeded *bool) filepath.WalkFunc {
	return func(path string, info os.FileInfo, err error) error {
		if err := fp.visit(path); err != nil {
			*changesNeeded = true
			if err != ErrChangesNeeded {
				return err
			}
		}
		return nil
	}
}

func wantedMunges() (filtered []munge) {
	skipList := strings.Split(*skipMunges, ",")
	skipped := map[string]bool{}
	for _, m := range skipList {
		if len(m) > 0 {
			skipped[m] = true
		}
	}
	for _, m := range allMunges {
		if !skipped[m.name] {
			filtered = append(filtered, m)
		} else {
			// Remove from the map so we can verify that everything
			// requested was in fact valid.
			delete(skipped, m.name)
		}
	}
	if len(skipped) != 0 {
		fmt.Fprintf(os.Stderr, "ERROR: requested to skip %v, but these are not valid munges. (valid: %v)\n", skipped, availableMungeList)
		os.Exit(1)
	}
	return filtered
}

func main() {
	flag.Parse()

	if *rootDir == "" {
		fmt.Fprintf(os.Stderr, "usage: %s [--verify] --root-dir <docs root>\n", flag.Arg(0))
		os.Exit(1)
	}

	// Split the root dir of "foo/docs" into "foo" and "docs". We
	// chdir into "foo" and walk "docs" so the walk is always at a
	// relative path.
	stem, leaf := path.Split(strings.TrimRight(*rootDir, "/"))
	if err := os.Chdir(stem); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(2)
	}

	fp := fileProcessor{
		munges:     wantedMunges(),
		verifyOnly: *verify,
	}

	// For each markdown file under source docs root, process the doc.
	// - If any error occurs: exit with failure (exit >1).
	// - If verify is true: exit 0 if no changes needed, exit 1 if changes
	//   needed.
	// - If verify is false: exit 0 if changes successfully made or no
	//   changes needed, exit 1 if manual changes are needed.
	var changesNeeded bool

	err := filepath.Walk(leaf, newWalkFunc(&fp, &changesNeeded))
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(2)
	}
	if changesNeeded {
		if *verify {
			fmt.Fprintf(os.Stderr, "FAIL: changes needed but not made due to --verify\n")
		} else {
			fmt.Fprintf(os.Stderr, "FAIL: some manual changes are still required.\n")
		}
		os.Exit(1)
	}
}
