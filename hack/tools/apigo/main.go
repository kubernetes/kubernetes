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
package main

import (
	"bufio"
	"flag"
	"fmt"
	"go/token"
	"go/types"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/tools/go/packages"
)

// use a well known location for the golang API changes
const apigoDir = "apigo"

func isInternalPackage(pkgPath string) bool {
	for _, part := range strings.Split(pkgPath, "/") {
		if part == "internal" {
			return true
		}
	}
	return false
}

// loadAPILines reads a text file and returns a set of non-empty, non-comment lines.
func loadAPILines(path string) map[string]bool {
	lines := make(map[string]bool)
	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return lines // Treat missing files as empty sets
		}
		log.Fatalf("Error reading %s: %v", path, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue // Skip empty lines and comments
		}
		lines[line] = true
	}
	return lines
}

// loadExceptions reads except.txt and strictly enforces that every exception
// is documented with a PR and Migration instruction in the preceding comment block.
func loadExceptions(path string) map[string]bool {
	exceptions := make(map[string]bool)
	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return exceptions // Return empty map if file doesn't exist
		}
		log.Fatalf("Error reading %s: %v", path, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var currentComments []string
	lineNumber := 0

	for scanner.Scan() {
		lineNumber++
		line := strings.TrimSpace(scanner.Text())

		if line == "" {
			// A blank line resets the comment context
			currentComments = nil
			continue
		}

		if strings.HasPrefix(line, "#") {
			// Accumulate comments
			currentComments = append(currentComments, line)
			continue
		}

		// If we reach here, it is an exception line. We must validate the comments.
		joinedComments := strings.ToUpper(strings.Join(currentComments, " "))
		hasPR := strings.Contains(joinedComments, "PR:")
		hasMigration := strings.Contains(joinedComments, "MIGRATION:")

		if !hasPR || !hasMigration {
			fmt.Printf("FORMAT ERROR in %s:%d\n", path, lineNumber)
			fmt.Printf("Exception: %s\n", line)
			fmt.Printf("Reason: The comment block immediately preceding this exception must contain 'PR:' and 'Migration:'.\n\n")
			os.Exit(1)
		}

		exceptions[line] = true
		// We do NOT reset currentComments here.
		// This allows multiple exceptions to share the same PR/Migration block.
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading %s: %v", path, err)
	}

	return exceptions
}

// extractCurrentAPI parses the Go code in the target directory and returns the exported API.
func extractCurrentAPI(targetDir string) map[string]bool {
	cfg := &packages.Config{
		Mode: packages.NeedName | packages.NeedTypes | packages.NeedImports | packages.NeedDeps,
		Dir:  targetDir,
	}

	pkgs, err := packages.Load(cfg, "./...")
	if err != nil {
		log.Fatalf("Failed to load packages in %s: %v", targetDir, err)
	}

	currentAPI := make(map[string]bool)

	for _, pkg := range pkgs {
		if pkg.Types == nil || isInternalPackage(pkg.PkgPath) {
			continue
		}

		qualifier := func(other *types.Package) string {
			if other == pkg.Types {
				return ""
			}
			return other.Path()
		}

		scope := pkg.Types.Scope()
		for _, name := range scope.Names() {
			if !token.IsExported(name) {
				continue
			}

			obj := scope.Lookup(name)
			pkgPath := pkg.PkgPath

			switch obj := obj.(type) {
			case *types.Const:
				currentAPI[fmt.Sprintf("pkg %s, const %s %s", pkgPath, name, types.TypeString(obj.Type(), qualifier))] = true
			case *types.Var:
				currentAPI[fmt.Sprintf("pkg %s, var %s %s", pkgPath, name, types.TypeString(obj.Type(), qualifier))] = true
			case *types.Func:
				sig := strings.TrimPrefix(types.TypeString(obj.Type(), qualifier), "func")
				currentAPI[fmt.Sprintf("pkg %s, func %s%s", pkgPath, name, sig)] = true
			case *types.TypeName:
				underlying := strings.ReplaceAll(types.TypeString(obj.Type().Underlying(), qualifier), "\n", " ")
				currentAPI[fmt.Sprintf("pkg %s, type %s %s", pkgPath, name, underlying)] = true

				if named, ok := obj.Type().(*types.Named); ok {
					for i := 0; i < named.NumMethods(); i++ {
						meth := named.Method(i)
						if token.IsExported(meth.Name()) {
							methSig := strings.TrimPrefix(types.TypeString(meth.Type(), qualifier), "func")
							currentAPI[fmt.Sprintf("pkg %s, method (%s) %s%s", pkgPath, name, meth.Name(), methSig)] = true
						}
					}
				}
			}
		}
	}
	return currentAPI
}

func main() {
	updateFlag := flag.Bool("update", false, "Update next.txt with new API additions")
	verifyFlag := flag.Bool("verify", false, "Verify that the current API matches the declared history")
	apiDirFlag := flag.String("api-dir", "", "Override the directory where .txt files are stored/loaded")
	syncExceptionsFlag := flag.Bool("sync-exceptions", false, "Auto-populate except.txt with missing historical symbols (bootstrap only)")
	flag.Parse()

	if !*updateFlag && !*verifyFlag && !*syncExceptionsFlag {
		log.Fatal("You must specify -update, -verify, or -sync-exceptions")
	}

	args := flag.Args()
	if len(args) < 1 {
		log.Fatal("You must provide the target module directory")
	}
	targetDir := args[0]
	// Default to targetDir/apigo, but allow the bash script to override it
	var apiDir string
	if *apiDirFlag != "" {
		apiDir = *apiDirFlag
	} else {
		apiDir = filepath.Join(targetDir, apigoDir)
	}
	// Extract Current API
	current := extractCurrentAPI(targetDir)

	// Load History (v*.txt)
	historical := make(map[string]bool)
	historyFiles, _ := filepath.Glob(filepath.Join(apiDir, "v*.txt"))
	for _, f := range historyFiles {
		for k := range loadAPILines(f) {
			historical[k] = true
		}
	}

	// Load Exceptions (except.txt)
	exceptions := loadExceptions(filepath.Join(apiDir, "except.txt"))

	// Load Next (next.txt)
	nextFile := filepath.Join(apiDir, "next.txt")
	next := loadAPILines(nextFile)

	hasErrors := false

	if *verifyFlag {
		// Check for missing items (Breaking Changes)
		// Rule: If it's in (History OR Next) and NOT in Exceptions, it MUST be in Current.
		for item := range historical {
			if !exceptions[item] && !current[item] {
				fmt.Printf("BREAKING CHANGE: %s\n   (It was removed but not added to except.txt)\n", item)
				hasErrors = true
			}
		}
		for item := range next {
			if !exceptions[item] && !current[item] {
				fmt.Printf("MISSING FROM CODE: %s\n   (It is listed in next.txt but doesn't exist in the code)\n", item)
				hasErrors = true
			}
		}

		// Check for unregistered additions
		// Rule: If it's in Current, it MUST be in (History OR Next).
		for item := range current {
			if !historical[item] && !next[item] {
				fmt.Printf("UNREGISTERED ADDITION: %s\n   (Run hack/update-apigo.sh to add it to next.txt)\n", item)
				hasErrors = true
			}
		}

		if hasErrors {
			os.Exit(1)
		}
	}

	if *updateFlag {
		var newAdditions []string

		// Everything in Current that is NOT in History belongs in next.txt
		for item := range current {
			if !historical[item] {
				newAdditions = append(newAdditions, item)
			}
		}
		sort.Strings(newAdditions)

		// Write to next.txt
		os.MkdirAll(apiDir, 0755)
		f, err := os.Create(nextFile)
		if err != nil {
			log.Fatalf("Failed to create %s: %v", nextFile, err)
		}
		for _, item := range newAdditions {
			fmt.Fprintln(f, item)
		}
		f.Close()
		fmt.Printf("Generated %s with %d new items.\n", nextFile, len(newAdditions))

		// Warn the user if they broke something so they don't get a surprise in CI
		for item := range historical {
			if !exceptions[item] && !current[item] {
				fmt.Printf("\n WARNING: You removed an exported symbol:\n   %s\n", item)
				fmt.Printf("   If this is intentional, you MUST manually add it to api/except.txt with a rationale.\n")
				hasErrors = true
			}
		}

		if hasErrors {
			os.Exit(1)
		}
	}

	if *syncExceptionsFlag {
		var missing []string
		for item := range historical {
			if !exceptions[item] && !current[item] {
				missing = append(missing, item)
			}
		}
		if len(missing) > 0 {
			sort.Strings(missing)
			f, err := os.OpenFile(filepath.Join(apiDir, "except.txt"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
			if err != nil {
				log.Fatalf("Failed to open except.txt: %v", err)
			}
			fmt.Fprintln(f, "\n# AUTO-GENERATED EXCEPTIONS FROM BOOTSTRAP")
			fmt.Fprintln(f, "# PR: Historical API Cleanups")
			fmt.Fprintln(f, "# MIGRATION: These APIs were removed in prior releases.")
			for _, item := range missing {
				fmt.Fprintln(f, item)
			}
			f.Close()
			fmt.Printf("Auto-logged %d historical exceptions to except.txt\n", len(missing))
		}
		os.Exit(0)
	}

}
