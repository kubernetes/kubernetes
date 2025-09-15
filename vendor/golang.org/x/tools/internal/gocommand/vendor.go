// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gocommand

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"golang.org/x/mod/semver"
)

// ModuleJSON holds information about a module.
type ModuleJSON struct {
	Path      string      // module path
	Version   string      // module version
	Versions  []string    // available module versions (with -versions)
	Replace   *ModuleJSON // replaced by this module
	Time      *time.Time  // time version was created
	Update    *ModuleJSON // available update, if any (with -u)
	Main      bool        // is this the main module?
	Indirect  bool        // is this module only an indirect dependency of main module?
	Dir       string      // directory holding files for this module, if any
	GoMod     string      // path to go.mod file used when loading this module, if any
	GoVersion string      // go version used in module
}

var modFlagRegexp = regexp.MustCompile(`-mod[ =](\w+)`)

// VendorEnabled reports whether vendoring is enabled. It takes a *Runner to execute Go commands
// with the supplied context.Context and Invocation. The Invocation can contain pre-defined fields,
// of which only Verb and Args are modified to run the appropriate Go command.
// Inspired by setDefaultBuildMod in modload/init.go
func VendorEnabled(ctx context.Context, inv Invocation, r *Runner) (bool, *ModuleJSON, error) {
	mainMod, go114, err := getMainModuleAnd114(ctx, inv, r)
	if err != nil {
		return false, nil, err
	}

	// We check the GOFLAGS to see if there is anything overridden or not.
	inv.Verb = "env"
	inv.Args = []string{"GOFLAGS"}
	stdout, err := r.Run(ctx, inv)
	if err != nil {
		return false, nil, err
	}
	goflags := string(bytes.TrimSpace(stdout.Bytes()))
	matches := modFlagRegexp.FindStringSubmatch(goflags)
	var modFlag string
	if len(matches) != 0 {
		modFlag = matches[1]
	}
	// Don't override an explicit '-mod=' argument.
	if modFlag == "vendor" {
		return true, mainMod, nil
	} else if modFlag != "" {
		return false, nil, nil
	}
	if mainMod == nil || !go114 {
		return false, nil, nil
	}
	// Check 1.14's automatic vendor mode.
	if fi, err := os.Stat(filepath.Join(mainMod.Dir, "vendor")); err == nil && fi.IsDir() {
		if mainMod.GoVersion != "" && semver.Compare("v"+mainMod.GoVersion, "v1.14") >= 0 {
			// The Go version is at least 1.14, and a vendor directory exists.
			// Set -mod=vendor by default.
			return true, mainMod, nil
		}
	}
	return false, nil, nil
}

// getMainModuleAnd114 gets one of the main modules' information and whether the
// go command in use is 1.14+. This is the information needed to figure out
// if vendoring should be enabled.
func getMainModuleAnd114(ctx context.Context, inv Invocation, r *Runner) (*ModuleJSON, bool, error) {
	const format = `{{.Path}}
{{.Dir}}
{{.GoMod}}
{{.GoVersion}}
{{range context.ReleaseTags}}{{if eq . "go1.14"}}{{.}}{{end}}{{end}}
`
	inv.Verb = "list"
	inv.Args = []string{"-m", "-f", format}
	stdout, err := r.Run(ctx, inv)
	if err != nil {
		return nil, false, err
	}

	lines := strings.Split(stdout.String(), "\n")
	if len(lines) < 5 {
		return nil, false, fmt.Errorf("unexpected stdout: %q", stdout.String())
	}
	mod := &ModuleJSON{
		Path:      lines[0],
		Dir:       lines[1],
		GoMod:     lines[2],
		GoVersion: lines[3],
		Main:      true,
	}
	return mod, lines[4] == "go1.14", nil
}

// WorkspaceVendorEnabled reports whether workspace vendoring is enabled. It takes a *Runner to execute Go commands
// with the supplied context.Context and Invocation. The Invocation can contain pre-defined fields,
// of which only Verb and Args are modified to run the appropriate Go command.
// Inspired by setDefaultBuildMod in modload/init.go
func WorkspaceVendorEnabled(ctx context.Context, inv Invocation, r *Runner) (bool, []*ModuleJSON, error) {
	inv.Verb = "env"
	inv.Args = []string{"GOWORK"}
	stdout, err := r.Run(ctx, inv)
	if err != nil {
		return false, nil, err
	}
	goWork := string(bytes.TrimSpace(stdout.Bytes()))
	if fi, err := os.Stat(filepath.Join(filepath.Dir(goWork), "vendor")); err == nil && fi.IsDir() {
		mainMods, err := getWorkspaceMainModules(ctx, inv, r)
		if err != nil {
			return false, nil, err
		}
		return true, mainMods, nil
	}
	return false, nil, nil
}

// getWorkspaceMainModules gets the main modules' information.
// This is the information needed to figure out if vendoring should be enabled.
func getWorkspaceMainModules(ctx context.Context, inv Invocation, r *Runner) ([]*ModuleJSON, error) {
	const format = `{{.Path}}
{{.Dir}}
{{.GoMod}}
{{.GoVersion}}
`
	inv.Verb = "list"
	inv.Args = []string{"-m", "-f", format}
	stdout, err := r.Run(ctx, inv)
	if err != nil {
		return nil, err
	}

	lines := strings.Split(strings.TrimSuffix(stdout.String(), "\n"), "\n")
	if len(lines) < 4 {
		return nil, fmt.Errorf("unexpected stdout: %q", stdout.String())
	}
	mods := make([]*ModuleJSON, 0, len(lines)/4)
	for i := 0; i < len(lines); i += 4 {
		mods = append(mods, &ModuleJSON{
			Path:      lines[i],
			Dir:       lines[i+1],
			GoMod:     lines[i+2],
			GoVersion: lines[i+3],
			Main:      true,
		})
	}
	return mods, nil
}
