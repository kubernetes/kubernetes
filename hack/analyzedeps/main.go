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
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
)

// getRepoRoot returns the absolute path to the root of the git repository.
func getRepoRoot() (string, error) {
	// Support KUBE_ROOT env var
	if val, ok := os.LookupEnv("KUBE_ROOT"); ok {
		return val, nil
	}
	// Try to find the repo root using git if invoked directly
	out, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		return "", fmt.Errorf("failed to find git repository root: %w", err)
	}
	return strings.TrimSpace(string(out)), nil
}

func main() {
	cacheDir := flag.String("cache-dir", "", "Optional directory to cache downloaded release tarballs")
	flag.Parse()

	args := flag.Args()
	if len(args) > 1 {
		fmt.Fprintln(os.Stderr, "Usage: analyzedeps [version|binaries-dir|local] [--cache-dir=<dir>]")
		flag.Usage()
		os.Exit(1)
	}

	target := "local"
	if len(args) == 1 && args[0] != "" {
		target = args[0]
	}

	// productionDeps tracks unique module import paths compiled into production binaries.
	var productionDeps map[string]bool
	var modInfo *GoModInfo
	var err error

	if target == "local" {
		productionDeps, err = scanLocalWorkspace()
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		goModPath, err := findGoMod()
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		modInfo, err = parseGoMod(goModPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
	} else if strings.HasPrefix(target, "v") {
		productionDeps, modInfo, err = processRemoteVersion(target, *cacheDir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
	} else {
		productionDeps, err = scanBinaries(target)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		goModPath, err := findGoMod()
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		modInfo, err = parseGoMod(goModPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
	}

	report, err := analyzeDependencies(productionDeps, modInfo)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	printReport(report)
}
