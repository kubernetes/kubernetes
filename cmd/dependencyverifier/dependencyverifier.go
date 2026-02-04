/*
Copyright 2021 The Kubernetes Authors.

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
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"sort"
	"strings"

	"github.com/google/go-cmp/cmp" //nolint:depguard
	"golang.org/x/tools/go/packages"
)

type Unwanted struct {
	// things we want to stop referencing
	Spec UnwantedSpec `json:"spec"`
	// status of our unwanted dependencies
	Status UnwantedStatus `json:"status"`
}

type UnwantedSpec struct {
	// module names we don't want to depend on, mapped to an optional message about why
	UnwantedModules map[string]string `json:"unwantedModules"`
	// module names that should never be updated from their current version, mapped to a struct with version and reason
	PinnedModules map[string]PinnedModule `json:"pinnedModules"`
}

type PinnedModule struct {
	Version string `json:"Version"`
	Reason  string `json:"Reason"`
}

// UnwantedReferenceInfo categorizes references to an unwanted module.
// This helps maintainers understand whether an unwanted dependency is actually used
// and where to focus remediation efforts.
type UnwantedReferenceInfo struct {
	// Direct lists modules that actually import this unwanted module in source.
	Direct []string `json:"direct,omitempty"`
	// Transitive lists modules that have this unwanted module in their dependency
	// graph or go.sum but do not import it in source. These modules are impacted by
	// the unwanted dependency but do not directly require source changes.
	Transitive []string `json:"transitive,omitempty"`
}

type UnwantedStatus struct {
	// references to modules in the spec.unwantedModules list, based on `go mod graph` content.
	// eliminating things from this list is good, and sometimes requires working with upstreams to do so.
	// References are categorized as "direct" (modules importing the unwanted dependency)
	// or "transitive" (modules that only carry it in their graph/go.sum).
	UnwantedReferences map[string]UnwantedReferenceInfo `json:"unwantedReferences"`
	// list of modules in the spec.unwantedModules list which are vendored
	UnwantedVendored []string `json:"unwantedVendored"`
}

// runCommand runs the cmd and returns the combined stdout and stderr, or an
// error if the command failed.
func runCommand(cmd ...string) (string, error) {
	return runCommandInDir("", cmd)
}

func runCommandInDir(dir string, cmd []string) (string, error) {
	return runCommandInDirWithEnv(dir, nil, cmd)
}

func runCommandInDirWithEnv(dir string, env []string, cmd []string) (string, error) {
	c := exec.Command(cmd[0], cmd[1:]...)
	c.Dir = dir
	c.Env = mergeEnv(os.Environ(), env...)
	output, err := c.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to run %q: %s (%s)", strings.Join(cmd, " "), err, output)
	}
	return string(output), nil
}

func mergeEnv(base []string, overrides ...string) []string {
	if len(overrides) == 0 {
		return append([]string(nil), base...)
	}
	merged := make([]string, 0, len(base)+len(overrides))
	seen := map[string]bool{}
	for _, override := range overrides {
		key := envKey(override)
		if key != "" {
			seen[key] = true
		}
	}
	for _, value := range base {
		key := envKey(value)
		if key != "" && seen[key] {
			continue
		}
		merged = append(merged, value)
	}
	merged = append(merged, overrides...)
	return merged
}

func envKey(env string) string {
	if idx := strings.Index(env, "="); idx != -1 {
		return env[:idx]
	}
	return ""
}

func readFile(path string) (string, error) {
	content, err := os.ReadFile(path)
	// Convert []byte to string and print to screen
	return string(content), err
}

func moduleInSlice(a module, list []module, matchVersion bool) bool {
	for _, b := range list {
		if b == a {
			return true
		}
		if !matchVersion && b.name == a.name {
			return true
		}
	}
	return false
}

// converts `go mod graph` output modStr into a map of from->[]to references and the main module
func convertToMap(modStr string) ([]module, map[module][]module) {
	var (
		mainModulesList = []module{}
		mainModules     = map[module]bool{}
	)
	modMap := make(map[module][]module)
	for _, line := range strings.Split(modStr, "\n") {
		if len(line) == 0 {
			continue
		}
		deps := strings.Split(line, " ")
		if len(deps) == 2 {
			first := parseModule(deps[0])
			second := parseModule(deps[1])
			if first.version == "" || first.version == "v0.0.0" {
				if !mainModules[first] {
					mainModules[first] = true
					mainModulesList = append(mainModulesList, first)
				}
			}
			modMap[first] = append(modMap[first], second)
		} else {
			// skip invalid line
			log.Printf("!!!invalid line in mod.graph: %s", line)
			continue
		}
	}
	return mainModulesList, modMap
}

// difference returns a-b and b-a as sorted lists
func difference(a, b []string) ([]string, []string) {
	aMinusB := map[string]bool{}
	bMinusA := map[string]bool{}
	for _, dependency := range a {
		aMinusB[dependency] = true
	}
	for _, dependency := range b {
		if _, found := aMinusB[dependency]; found {
			delete(aMinusB, dependency)
		} else {
			bMinusA[dependency] = true
		}
	}
	aMinusBList := []string{}
	bMinusAList := []string{}
	for dependency := range aMinusB {
		aMinusBList = append(aMinusBList, dependency)
	}
	for dependency := range bMinusA {
		bMinusAList = append(bMinusAList, dependency)
	}
	sort.Strings(aMinusBList)
	sort.Strings(bMinusAList)
	return aMinusBList, bMinusAList
}

type module struct {
	name    string
	version string
}

type targetPlatform struct {
	goos   string
	goarch string
}

func parsePlatforms(value string) ([]targetPlatform, error) {
	if strings.TrimSpace(value) == "" {
		return nil, nil
	}
	entries := strings.Split(value, ",")
	platforms := make([]targetPlatform, 0, len(entries))
	for _, entry := range entries {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		parts := strings.SplitN(entry, "/", 2)
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			return nil, fmt.Errorf("invalid platform %q, expected goos/goarch", entry)
		}
		platforms = append(platforms, targetPlatform{goos: parts[0], goarch: parts[1]})
	}
	return platforms, nil
}

func formatTargets(targets []targetPlatform) string {
	parts := make([]string, 0, len(targets))
	for _, target := range targets {
		parts = append(parts, target.goos+"/"+target.goarch)
	}
	return strings.Join(parts, ",")
}

func (m module) String() string {
	if len(m.version) == 0 {
		return m.name
	}
	return m.name + "@" + m.version
}

func parseModule(s string) module {
	if !strings.Contains(s, "@") {
		return module{name: s}
	}
	parts := strings.SplitN(s, "@", 2)
	return module{name: parts[0], version: parts[1]}
}

// goPackagesImportsByTarget uses go/packages to load all packages in dir across multiple
// target platforms and returns the union of all imported modules.
// -buildvcs=false is needed because module cache is read-only without VCS info.
// Tests and the default build tags are included to mirror Kubernetes' typecheck behavior.
func goPackagesImportsByTarget(dir string, targets []targetPlatform, skipTests bool, buildTags []string, debugModules map[string]bool) (map[string]bool, error) {
	imports := map[string]bool{}
	var errs []error
	successes := 0

	for _, target := range targets {
		moduleImports, err := goPackagesImportsInDir(dir, target, skipTests, buildTags, debugModules)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		successes++
		for impPath := range moduleImports {
			imports[impPath] = true
		}
	}

	if successes == 0 {
		return nil, fmt.Errorf("go/packages load failed for all target platforms: %w", errors.Join(errs...))
	}

	return imports, nil
}

func goPackagesImportsInDir(dir string, target targetPlatform, skipTests bool, buildTags []string, debugModules map[string]bool) (map[string]bool, error) {
	imports := map[string]bool{}
	env := []string{
		"GOOS=" + target.goos,
		"GOARCH=" + target.goarch,
		"CGO_ENABLED=0",
	}
	if len(buildTags) == 0 {
		buildTags = []string{"selinux"}
	}
	config := &packages.Config{
		Mode:       packages.NeedName | packages.NeedImports | packages.NeedModule | packages.NeedFiles,
		Dir:        dir,
		Env:        mergeEnv(os.Environ(), env...),
		BuildFlags: []string{"-buildvcs=false", "-tags", strings.Join(buildTags, ",")},
		Tests:      !skipTests,
	}
	pkgs, err := packages.Load(config, "./...")
	if err != nil {
		return nil, fmt.Errorf("%s/%s: %w", target.goos, target.goarch, err)
	}
	if pkgErrs := collectPackageErrors(pkgs); len(pkgErrs) > 0 {
		return nil, fmt.Errorf("%s/%s: %w", target.goos, target.goarch, errors.Join(pkgErrs...))
	}
	for _, pkg := range pkgs {
		if skipTests && len(pkg.GoFiles) == 0 {
			continue
		}
		importerModule := modulePath(pkg.Module)
		for _, impPkg := range pkg.Imports {
			if impPkg == nil || impPkg.Module == nil {
				continue
			}
			imports[impPkg.Module.Path] = true
			if len(debugModules) > 0 && (debugModules[importerModule] || debugModules[impPkg.Module.Path]) {
				log.Printf(
					"debug-imports target=%s/%s importerModule=%s importerPkg=%s importedModule=%s",
					target.goos,
					target.goarch,
					importerModule,
					pkg.PkgPath,
					impPkg.Module.Path,
				)
			}
		}
	}
	return imports, nil
}

func collectPackageErrors(pkgs []*packages.Package) []error {
	var errs []error
	for _, pkg := range pkgs {
		for _, pkgErr := range pkg.Errors {
			errs = append(errs, errors.New(pkgErr.Error()))
		}
	}
	return errs
}

// buildModuleImportsMap downloads each module and runs go/packages from within
// the module directory to determine which modules it actually imports.
// Returns a map of module name -> set of module names it imports.
func buildModuleImportsMap(modulesToCheck []string, moduleVersions map[string]string, skipTests bool, buildTags []string, targets []targetPlatform, debugModules map[string]bool) (map[string]map[string]bool, error) {
	if len(modulesToCheck) == 0 {
		return make(map[string]map[string]bool), nil
	}
	if len(targets) == 0 {
		targets = []targetPlatform{
			{goos: "linux", goarch: "amd64"},
			{goos: "linux", goarch: "arm64"},
		}
	}

	moduleImports := make(map[string]map[string]bool)
	for _, mod := range modulesToCheck {
		version := moduleVersions[mod]
		if version == "" || version == "v0.0.0" {
			continue
		}
		log.Printf("dependencyverifier: analyze module %s@%s", mod, version)
		// Download the module and get its directory using go mod download -json
		output, err := runCommand("go", "mod", "download", "-json", mod+"@"+version)
		if err != nil {
			// Module might not be downloadable, skip it
			continue
		}

		// Parse the JSON to get the Dir field
		var downloadInfo struct {
			Dir string `json:"Dir"`
		}
		if err := json.Unmarshal([]byte(output), &downloadInfo); err != nil {
			continue
		}
		if downloadInfo.Dir == "" {
			continue
		}

		importPaths, err := goPackagesImportsByTarget(downloadInfo.Dir, targets, skipTests, buildTags, debugModules)
		if err != nil {
			// Module might have replace directives with relative paths that don't work.
			// Try copying to a temp dir and removing replace directives.
			importPaths, err = runGoPackagesWithoutReplace(downloadInfo.Dir, targets, skipTests, buildTags, debugModules)
			if err != nil {
				// Still failed, skip it
				continue
			}
		}

		moduleImports[mod] = make(map[string]bool)
		for impModule := range importPaths {
			if impModule != mod {
				moduleImports[mod][impModule] = true
			}
		}
	}

	return moduleImports, nil
}

// runGoPackagesWithoutReplace copies a module to a temp directory, removes replace
// directives from go.mod, and runs go/packages. This handles modules like etcd that
// use replace directives with relative paths that don't work when downloaded alone.
func runGoPackagesWithoutReplace(moduleDir string, targets []targetPlatform, skipTests bool, buildTags []string, debugModules map[string]bool) (map[string]bool, error) {
	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "depverifier-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)

	// Copy module contents to temp dir.
	if err := os.CopyFS(tmpDir, os.DirFS(moduleDir)); err != nil {
		return nil, err
	}

	// Make go.mod and go.sum writable (module cache files are read-only)
	goModPath := tmpDir + "/go.mod"
	if err := os.Chmod(goModPath, 0644); err != nil {
		return nil, err
	}
	goSumPath := tmpDir + "/go.sum"
	if _, err := os.Stat(goSumPath); err == nil {
		if err := os.Chmod(goSumPath, 0644); err != nil {
			return nil, err
		}
	}
	goModContent, err := os.ReadFile(goModPath)
	if err != nil {
		return nil, err
	}

	// Remove replace blocks and single replace directives
	lines := strings.Split(string(goModContent), "\n")
	var newLines []string
	inReplaceBlock := false
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "replace (") || strings.HasPrefix(trimmed, "replace(") {
			inReplaceBlock = true
			continue
		}
		if inReplaceBlock {
			if trimmed == ")" {
				inReplaceBlock = false
			}
			continue
		}
		if strings.HasPrefix(trimmed, "replace ") {
			continue
		}
		newLines = append(newLines, line)
	}

	if err := os.WriteFile(goModPath, []byte(strings.Join(newLines, "\n")), 0644); err != nil {
		return nil, err
	}

	// Update go.sum after removing replace directives
	if _, err := runCommandInDir(tmpDir, []string{"go", "mod", "tidy"}); err != nil {
		return nil, err
	}

	// Run go/packages in the temp directory
	return goPackagesImportsByTarget(tmpDir, targets, skipTests, buildTags, debugModules)
}

func modulePath(mod *packages.Module) string {
	if mod == nil {
		return ""
	}
	return mod.Path
}

// isDirectImporter checks if a module actually imports the unwanted module in its source code.
// It uses the pre-computed moduleImports map from go/packages analysis.
// Returns true if the module has actual import statements for the unwanted module.
func isDirectImporter(moduleImports map[string]map[string]bool, moduleName, unwantedModule string) bool {
	imports, ok := moduleImports[moduleName]
	if !ok {
		// Module not found in imports map (go/packages may have failed) - assume no direct imports
		return false
	}
	if imports[unwantedModule] {
		return true
	}
	for importedModule := range imports {
		if strings.HasPrefix(importedModule, unwantedModule+"/") {
			return true
		}
	}
	return false
}

// option1: dependencyverifier dependencies.json
// it will run `go mod graph` and check it.
func main() {
	skipTests := flag.Bool("skip-test", false, "exclude test files when computing module imports")
	buildTagsFlag := flag.String("tags", "selinux,weak_dependency", "comma-separated build tags to pass to go/packages")
	platformsFlag := flag.String("platforms", "", "comma-separated target platforms (goos/goarch) to check")
	cross := flag.Bool("cross", false, "check across common GOOS/GOARCH targets")
	debugImports := flag.String("debug-imports", "", "comma-separated module paths to log import sources for")
	flag.Parse()

	var modeGraphStr string
	var err error
	if flag.NArg() == 1 {
		// run `go mod graph`
		modeGraphStr, err = runCommand("go", "mod", "graph")
		if err != nil {
			log.Fatalf("Error running 'go mod graph': %s", err)
		}
	} else {
		log.Fatalf("Usage: %s [--skip-test] dependencies.json", os.Args[0])
	}

	dependenciesJSONPath := flag.Arg(0)
	dependencies, err := readFile(dependenciesJSONPath)
	if err != nil {
		log.Fatalf("Error reading dependencies file %s: %s", dependencies, err)
	}

	// load Unwanted from json
	configFromFile := &Unwanted{}
	decoder := json.NewDecoder(bytes.NewBuffer([]byte(dependencies)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(configFromFile); err != nil {
		log.Fatalf("Error reading dependencies file %s: %s", dependenciesJSONPath, err)
	}

	// convert from `go mod graph` to main module and map of from->[]to references
	mainModules, moduleGraph := convertToMap(modeGraphStr)

	directDependencies := map[string]map[string]bool{}
	for _, mainModule := range mainModules {
		dir := ""
		if mainModule.name != "k8s.io/kubernetes" {
			dir = "staging/src/" + mainModule.name
		}
		listOutput, err := runCommandInDir(dir, []string{"go", "list", "-m", "-buildvcs=false", "-f", "{{if not .Indirect}}{{if not .Main}}{{.Path}}{{end}}{{end}}", "all"})
		if err != nil {
			log.Fatalf("Error running go/packages for %s: %s", mainModule.name, err)
		}
		directDependencies[mainModule.name] = map[string]bool{}
		for _, directDependency := range strings.Split(listOutput, "\n") {
			directDependencies[mainModule.name][directDependency] = true
		}
	}

	// gather the effective versions by looking at the versions required by the main modules
	effectiveVersions := map[string]module{}
	for _, mainModule := range mainModules {
		for _, override := range moduleGraph[mainModule] {
			if _, ok := effectiveVersions[override.name]; !ok {
				effectiveVersions[override.name] = override
			}
		}
	}

	// Convert effectiveVersions to simple map[string]string for module versions
	// Include ALL modules from the graph so we can detect imports of transitive deps (like unwanted modules)
	moduleVersions := make(map[string]string)
	for name, mod := range effectiveVersions {
		moduleVersions[name] = mod.version
	}
	// Also include all modules from the full graph (not just direct deps of main modules)
	// This ensures we can detect imports of unwanted modules that are transitive dependencies
	for from, tos := range moduleGraph {
		if from.version != "" && moduleVersions[from.name] == "" {
			moduleVersions[from.name] = from.version
		}
		for _, to := range tos {
			if to.version != "" && moduleVersions[to.name] == "" {
				moduleVersions[to.name] = to.version
			}
		}
	}

	// Check for pinned modules that have been updated
	pinnedModuleViolations := map[string][]string{}
	if len(configFromFile.Spec.PinnedModules) > 0 {
		// Get the current versions of pinned modules
		for pinnedModule, pinnedInfo := range configFromFile.Spec.PinnedModules {
			// Check if the module is in effectiveVersions
			if effectiveModule, ok := effectiveVersions[pinnedModule]; ok {
				// Compare with the pinned version from the JSON file
				if effectiveModule.version != pinnedInfo.Version {
					pinnedModuleViolations[pinnedModule] = []string{
						fmt.Sprintf("Pinned version: %s", pinnedInfo.Version),
						fmt.Sprintf("Attempted update to: %s", effectiveModule.version),
						fmt.Sprintf("Reason for pinning: %s", pinnedInfo.Reason),
					}
				}
			}
		}
	}

	unwantedToReferencers := map[string][]module{}
	for _, mainModule := range mainModules {
		// visit to find unwanted modules still referenced from the main module
		visit(func(m module, via []module) {
			if _, unwanted := configFromFile.Spec.UnwantedModules[m.name]; unwanted {
				// this is unwanted, store what is referencing it
				referencer := via[len(via)-1]
				if !moduleInSlice(referencer, unwantedToReferencers[m.name], false) {
					// // uncomment to get a detailed tree of the path that referenced the unwanted dependency
					//
					// i := 0
					// for _, v := range via {
					// 	if v.version != "" && v.version != "v0.0.0" {
					// 		fmt.Println(strings.Repeat("  ", i), v)
					// 		i++
					// 	}
					// }
					// if i > 0 {
					// 	fmt.Println(strings.Repeat("  ", i+1), m)
					// 	fmt.Println()
					// }
					unwantedToReferencers[m.name] = append(unwantedToReferencers[m.name], referencer)
				}
			}
		}, mainModule, moduleGraph, effectiveVersions)
	}

	// Collect all third-party modules that reference unwanted modules
	// so we can batch-check which ones actually import the unwanted modules
	modulesToCheck := make(map[string]bool)
	for _, referencers := range unwantedToReferencers {
		for _, referencer := range referencers {
			if referencer.version != "" && referencer.version != "v0.0.0" {
				modulesToCheck[referencer.name] = true
			}
		}
	}
	modulesToCheckList := make([]string, 0, len(modulesToCheck))
	for mod := range modulesToCheck {
		modulesToCheckList = append(modulesToCheckList, mod)
	}

	// Build module imports map using go/packages for accurate detection
	debugModules := map[string]bool{}
	if *debugImports != "" {
		for _, mod := range strings.Split(*debugImports, ",") {
			if mod == "" {
				continue
			}
			debugModules[strings.TrimSpace(mod)] = true
		}
	}
	buildTags := []string{}
	if *buildTagsFlag != "" {
		for _, tag := range strings.Split(*buildTagsFlag, ",") {
			tag = strings.TrimSpace(tag)
			if tag != "" {
				buildTags = append(buildTags, tag)
			}
		}
	}
	var targets []targetPlatform
	if *cross {
		targets = []targetPlatform{
			{goos: "linux", goarch: "amd64"},
			{goos: "windows", goarch: "386"},
			{goos: "darwin", goarch: "amd64"},
			{goos: "darwin", goarch: "arm64"},
			{goos: "linux", goarch: "arm"},
			{goos: "linux", goarch: "386"},
			{goos: "windows", goarch: "amd64"},
			{goos: "linux", goarch: "arm64"},
			{goos: "linux", goarch: "ppc64le"},
			{goos: "linux", goarch: "s390x"},
			{goos: "windows", goarch: "arm64"},
		}
	}
	if len(targets) == 0 {
		parsed, err := parsePlatforms(*platformsFlag)
		if err != nil {
			log.Fatalf("Error parsing --platforms: %v", err)
		}
		targets = parsed
	}
	log.Printf("dependencyverifier: modules=%d skipTests=%t tags=%q targets=%q", len(modulesToCheckList), *skipTests, strings.Join(buildTags, ","), formatTargets(targets))
	moduleImports, err := buildModuleImportsMap(modulesToCheckList, moduleVersions, *skipTests, buildTags, targets, debugModules)
	if err != nil {
		log.Fatalf("Error building module imports map: %s", err)
	}

	config := &Unwanted{}
	config.Spec.UnwantedModules = configFromFile.Spec.UnwantedModules
	config.Status.UnwantedReferences = map[string]UnwantedReferenceInfo{}
	for unwanted := range unwantedToReferencers {
		sort.Slice(unwantedToReferencers[unwanted], func(i, j int) bool {
			ri := unwantedToReferencers[unwanted][i]
			rj := unwantedToReferencers[unwanted][j]
			if ri.name != rj.name {
				return ri.name < rj.name
			}
			return ri.version < rj.version
		})
		refInfo := UnwantedReferenceInfo{}
		for _, referencer := range unwantedToReferencers[unwanted] {
			// record specific names of versioned referents (third-party modules)
			if referencer.version != "" && referencer.version != "v0.0.0" {
				// Check if this module actually imports the unwanted package
				// or just has it in go.mod because of its own dependencies
				if isDirectImporter(moduleImports, referencer.name, unwanted) {
					refInfo.Direct = append(refInfo.Direct, referencer.name)
				} else {
					refInfo.Transitive = append(refInfo.Transitive, referencer.name)
				}
			} else if directDependencies[referencer.name][unwanted] {
				// main modules that directly depend on the unwanted module
				refInfo.Direct = append(refInfo.Direct, referencer.name)
			}
		}
		// only add entry if there are actual references
		if len(refInfo.Direct) > 0 || len(refInfo.Transitive) > 0 {
			config.Status.UnwantedReferences[unwanted] = refInfo
		}
	}

	vendorModulesTxt, err := os.ReadFile("vendor/modules.txt")
	if err != nil {
		log.Fatal(err)
	}
	vendoredModules := map[string]bool{}
	for _, l := range strings.Split(string(vendorModulesTxt), "\n") {
		parts := strings.Split(l, " ")
		if len(parts) == 3 && parts[0] == "#" && strings.HasPrefix(parts[2], "v") {
			vendoredModules[parts[1]] = true
		}
	}
	config.Status.UnwantedVendored = []string{}
	for unwanted := range configFromFile.Spec.UnwantedModules {
		if vendoredModules[unwanted] {
			config.Status.UnwantedVendored = append(config.Status.UnwantedVendored, unwanted)
		}
	}
	sort.Strings(config.Status.UnwantedVendored)

	needUpdate := false

	// Compare unwanted list from unwanted-dependencies.json with current status from `go mod graph`
	expected, err := json.MarshalIndent(configFromFile.Status, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	actual, err := json.MarshalIndent(config.Status, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if !bytes.Equal(expected, actual) {
		log.Printf("Expected status of\n%s", string(expected))
		log.Printf("Got status of\n%s", string(actual))
		needUpdate = true
		log.Print("Status diff:\n", cmp.Diff(expected, actual))
	}
	for expectedRef, expectedFrom := range configFromFile.Status.UnwantedReferences {
		actualFrom, ok := config.Status.UnwantedReferences[expectedRef]
		if !ok {
			// disappeared entirely
			log.Printf("Good news! Unwanted dependency %q is no longer referenced. Remove status.unwantedReferences[%q] in %s to ensure it doesn't get reintroduced.", expectedRef, expectedRef, dependenciesJSONPath)
			needUpdate = true
			continue
		}
		// Check direct references
		removedDirect, addedDirect := difference(expectedFrom.Direct, actualFrom.Direct)
		if len(removedDirect) > 0 {
			log.Printf("Good news! Unwanted module %q dropped the following direct dependants:", expectedRef)
			for _, reference := range removedDirect {
				log.Printf("   %s (direct)", reference)
			}
			log.Printf("!!! Remove those from status.unwantedReferences[%q].direct in %s to ensure they don't get reintroduced.", expectedRef, dependenciesJSONPath)
			needUpdate = true
		}
		if len(addedDirect) > 0 {
			log.Printf("Unwanted module %q marked in %s is referenced by new direct dependants:", expectedRef, dependenciesJSONPath)
			for _, reference := range addedDirect {
				log.Printf("   %s (direct)", reference)
			}
			log.Printf("!!! Avoid adding direct dependencies on unwanted modules\n")
			needUpdate = true
		}
		// Check transitive references (modules with dependency-only references)
		removedTransitive, addedTransitive := difference(expectedFrom.Transitive, actualFrom.Transitive)
		if len(removedTransitive) > 0 {
			log.Printf("Good news! Unwanted module %q dropped the following transitive dependants:", expectedRef)
			for _, reference := range removedTransitive {
				log.Printf("   %s (transitive)", reference)
			}
			log.Printf("!!! Remove those from status.unwantedReferences[%q].transitive in %s to ensure they don't get reintroduced.", expectedRef, dependenciesJSONPath)
			needUpdate = true
		}
		if len(addedTransitive) > 0 {
			log.Printf("Unwanted module %q marked in %s is referenced by new transitive dependants:", expectedRef, dependenciesJSONPath)
			for _, reference := range addedTransitive {
				log.Printf("   %s (transitive - does not import, just in go.sum/graph)", reference)
			}
			log.Printf("!!! Avoid updating referencing modules to versions that reintroduce unwanted dependencies\n")
			needUpdate = true
		}
	}
	for actualRef, actualFrom := range config.Status.UnwantedReferences {
		if _, expected := configFromFile.Status.UnwantedReferences[actualRef]; expected {
			// expected, already ensured referencers were equal in the first loop
			continue
		}
		log.Printf("Unwanted module %q marked in %s is referenced", actualRef, dependenciesJSONPath)
		for _, reference := range actualFrom.Direct {
			log.Printf("   %s (direct)", reference)
		}
		for _, reference := range actualFrom.Transitive {
			log.Printf("   %s (transitive - doesn't import, just in go.sum/graph)", reference)
		}
		log.Printf("!!! Avoid updating referencing modules to versions that reintroduce use of unwanted dependencies\n")
		needUpdate = true
	}

	removedVendored, addedVendored := difference(configFromFile.Status.UnwantedVendored, config.Status.UnwantedVendored)
	if len(removedVendored) > 0 {
		log.Printf("Good news! Unwanted modules are no longer vendered: %q", removedVendored)
		log.Printf("!!! Remove those from status.unwantedVendored in %s to ensure they don't get reintroduced.", dependenciesJSONPath)
		needUpdate = true
	}
	if len(addedVendored) > 0 {
		log.Printf("Unwanted modules are newly vendored: %q", addedVendored)
		log.Printf("!!! Avoid updates that increase vendoring of unwanted dependencies\n")
		needUpdate = true
	}

	if needUpdate {
		os.Exit(1)
	}

	// Check if there are any pinned module violations
	if len(pinnedModuleViolations) > 0 {
		log.Printf("ERROR: The following pinned modules have been updated:")
		for module, details := range pinnedModuleViolations {
			log.Printf("Module: %s", module)
			for _, detail := range details {
				log.Printf("  %s", detail)
			}
		}
		log.Printf("Pinned modules must not be updated. Please revert these changes.")
		os.Exit(1)
	}
}

func visit(visitor func(m module, via []module), main module, references map[module][]module, effectiveVersions map[string]module) {
	doVisit(visitor, main, nil, map[module]bool{}, references, effectiveVersions)
}

func doVisit(visitor func(m module, via []module), from module, via []module, visited map[module]bool, references map[module][]module, effectiveVersions map[string]module) {
	visitor(from, via)
	via = append(via, from)
	if visited[from] {
		return
	}
	for _, to := range references[from] {
		// switch to the effective version of this dependency
		if override, ok := effectiveVersions[to.name]; ok {
			to = override
		}
		// recurse unless we've already visited this module in this traversal
		if !moduleInSlice(to, via, false) {
			doVisit(visitor, to, via, visited, references, effectiveVersions)
		}
	}
	visited[from] = true
}
