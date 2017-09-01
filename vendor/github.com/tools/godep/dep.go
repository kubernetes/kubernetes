package main

import (
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strings"
)

// A Dependency is a specific revision of a package.
type Dependency struct {
	ImportPath string
	Comment    string `json:",omitempty"` // Description of commit, if present.
	Rev        string // VCS-specific commit ID.

	// used by command save & update
	ws   string // workspace
	root string // import path to repo root
	dir  string // full path to package

	// used by command update
	matched bool // selected for update by command line
	pkg     *Package
	missing bool // packages is missing

	// used by command go
	vcs *VCS
}

func eqDeps(a, b []Dependency) bool {
	ok := true
	for _, da := range a {
		for _, db := range b {
			if da.ImportPath == db.ImportPath && da.Rev != db.Rev {
				ok = false
			}
		}
	}
	return ok
}

// containsPathPrefix returns whether any string in a
// is s or a directory containing s.
// For example, pattern ["a"] matches "a" and "a/b"
// (but not "ab").
func containsPathPrefix(pats []string, s string) bool {
	for _, pat := range pats {
		if pat == s || strings.HasPrefix(s, pat+"/") {
			return true
		}
	}
	return false
}

func uniq(a []string) []string {
	var s string
	var i int
	if !sort.StringsAreSorted(a) {
		sort.Strings(a)
	}
	for _, t := range a {
		if t != s {
			a[i] = t
			i++
			s = t
		}
	}
	return a[:i]
}

// trimGoVersion and return the major version
func trimGoVersion(version string) (string, error) {
	if version == "devel" {
		return "devel", nil
	}
	if strings.HasPrefix(version, "devel+") || strings.HasPrefix(version, "devel-") {
		return strings.Replace(version, "devel+", "devel-", 1), nil
	}
	p := strings.Split(version, ".")
	if len(p) < 2 {
		return "", fmt.Errorf("Error determining major go version from: %q", version)
	}
	var split string
	switch {
	case strings.Contains(p[1], "beta"):
		split = "beta"
	case strings.Contains(p[1], "rc"):
		split = "rc"
	}
	if split != "" {
		p[1] = strings.Split(p[1], split)[0]
	}
	return p[0] + "." + p[1], nil
}

var goVersionTestOutput = ""

func getGoVersion() (string, error) {
	// For testing purposes only
	if goVersionTestOutput != "" {
		return goVersionTestOutput, nil
	}

	// Godep might have been compiled with a different
	// version, so we can't just use runtime.Version here.
	cmd := exec.Command("go", "version")
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	return string(out), err
}

// goVersion returns the major version string of the Go compiler
// currently installed, e.g. "go1.5".
func goVersion() (string, error) {
	out, err := getGoVersion()
	if err != nil {
		return "", err
	}
	gv := strings.Split(out, " ")
	if len(gv) < 4 {
		return "", fmt.Errorf("Error splitting output of `go version`: Expected 4 or more elements, but there are < 4: %q", out)
	}
	if gv[2] == "devel" {
		return trimGoVersion(gv[2] + gv[3])
	}
	return trimGoVersion(gv[2])
}
