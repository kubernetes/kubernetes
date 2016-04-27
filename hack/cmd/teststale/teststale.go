/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

var (
	binary  = flag.String("binary", "", "absolute filesystem path to the test binary")
	pkgPath = flag.String("package", "", "test package import path in the format used in the import statements without the $GOPATH prefix")
)

type pkg struct {
	dir          string
	target       string
	stale        bool
	testGoFiles  []string
	testImports  []string
	xTestGoFiles []string
	xTestImports []string
}

func newCmd(format string, pkgPaths []string) *exec.Cmd {
	args := []string{
		"list",
		"-f",
		format,
	}
	args = append(args, pkgPaths...)
	cmd := exec.Command("go", args...)
	cmd.Env = os.Environ()
	return cmd
}

func newPkg(path string) (*pkg, error) {
	format := "Dir: {{println .Dir}}Target: {{println .Target}}Stale: {{println .Stale}}TestGoFiles: {{println .TestGoFiles}}TestImports: {{println .TestImports}}XTestGoFiles: {{println .XTestGoFiles}}XTestImports: {{println .XTestImports}}"
	cmd := newCmd(format, []string{path})
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("could not pipe STDOUT: %v", err)
	}

	// Start executing the command
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("command did not start: %v", err)
	}

	// Parse the command output
	scanner := bufio.NewScanner(stdout)
	scanner.Split(bufio.ScanLines)

	// To be conservative, default to package to be stale
	p := &pkg{
		stale: true,
	}

	// TODO: avoid this stupid code repetition by iterating through struct fields.
	scanner.Scan()
	p.dir = strings.TrimPrefix(scanner.Text(), "Dir: ")
	scanner.Scan()
	p.target = strings.TrimPrefix(scanner.Text(), "Target: ")
	scanner.Scan()
	if strings.TrimPrefix(scanner.Text(), "Stale: ") == "false" {
		p.stale = false
	}
	p.testGoFiles = scanLineList(scanner, "TestGoFiles: ")
	p.testImports = scanLineList(scanner, "TestImports: ")
	p.xTestGoFiles = scanLineList(scanner, "XTestGoFiles: ")
	p.xTestImports = scanLineList(scanner, "XTestImports: ")

	if err := cmd.Wait(); err != nil {
		return nil, fmt.Errorf("command did not complete: %v", err)
	}
	return p, nil
}

func (p *pkg) isStale(buildTime time.Time) bool {
	// If the package itself is stale, then we have to rebuild the whole thing anyway.
	if p.stale {
		return true
	}

	// Test for file staleness
	for _, f := range p.testGoFiles {
		if isStale(buildTime, filepath.Join(p.dir, f)) {
			log.Printf("test Go file %s is stale", f)
			return true
		}
	}
	for _, f := range p.xTestGoFiles {
		if isStale(buildTime, filepath.Join(p.dir, f)) {
			log.Printf("external test Go file %s is stale", f)
			return true
		}
	}

	format := "{{.Stale}}"
	imps := []string{}
	imps = append(imps, p.testImports...)
	imps = append(imps, p.xTestImports...)

	cmd := newCmd(format, imps)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Printf("unexpected error with creating stdout pipe: %v", err)
		return true
	}
	// Start executing the command
	if err := cmd.Start(); err != nil {
		log.Printf("unexpected error executing command: %v", err)
		return true
	}

	// Parse the command output
	scanner := bufio.NewScanner(stdout)
	scanner.Split(bufio.ScanLines)

	for i := 0; scanner.Scan(); i++ {
		if out := scanner.Text(); out != "false" {
			log.Printf("import %q is stale: %s", imps[i], out)
			return true
		}
	}

	if err := cmd.Wait(); err != nil {
		log.Printf("unexpected error waiting to finish: %v", err)
		return true
	}
	return false
}

// scanLineList scans a line, removes the prefix and splits the remaining line into
// individual strings.
// TODO: There are ton of intermediate strings being created here. Convert this to
// a bufio.SplitFunc instead.
func scanLineList(scanner *bufio.Scanner, prefix string) []string {
	scanner.Scan()
	list := strings.TrimPrefix(scanner.Text(), prefix)
	line := strings.Trim(list, "[]")
	if len(line) == 0 {
		return []string{}
	}
	return strings.Split(line, " ")
}

func isStale(buildTime time.Time, filename string) bool {
	stat, err := os.Stat(filename)
	if err != nil {
		return true
	}
	return stat.ModTime().After(buildTime)
}

// IsTestStale checks if the test binary is stale and needs to rebuilt.
// Some of the ideas here are inspired by how Go does staleness checks.
func isTestStale(binPath, pkgPath string) bool {
	bStat, err := os.Stat(binPath)
	if err != nil {
		log.Printf("Couldn't obtain the modified time of the binary: %v", err)
		return true
	}
	buildTime := bStat.ModTime()

	p, err := newPkg(pkgPath)
	if err != nil {
		log.Printf("Couldn't retrieve the test package information: %v", err)
		return false
	}

	return p.isStale(buildTime)
}

func main() {
	flag.Parse()
	if isTestStale(*binary, *pkgPath) {
		fmt.Println("true")
	} else {
		fmt.Println("false")
	}
}
