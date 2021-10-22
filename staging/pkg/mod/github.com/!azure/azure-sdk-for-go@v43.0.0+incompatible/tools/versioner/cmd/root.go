// Copyright 2018 Microsoft Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cmd

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/Azure/azure-sdk-for-go/tools/apidiff/repo"
	"github.com/Azure/azure-sdk-for-go/tools/internal/modinfo"
	"github.com/Masterminds/semver"
	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "versioner <staging dir> [initial module version]",
	Short: "Creates or updates the latest major version for a package from staged content.",
	Long: `This tool will compare a staged package against its latest major version to detect
breaking changes.  If there are no breaking changes the latest major version is updated
with the staged content.  If there are breaking changes the staged content becomes the
next latest major vesion and the go.mod file is updated.
The default version for new modules is v1.0.0 or the value specified for [initial module version].
`,
	Args: func(cmd *cobra.Command, args []string) error {
		if err := cobra.MinimumNArgs(1)(cmd, args); err != nil {
			return err
		}
		if err := cobra.MaximumNArgs(2)(cmd, args); err != nil {
			return err
		}
		return nil
	},
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		return theCommand(args)
	},
}

var (
	semverRegex = regexp.MustCompile(`v\d+\.\d+\.\d+$`)
	// this is used so tests can hook getTags() to return whatever tags
	getTagsHook func(string, string) ([]string, error)
	// default version to start a module at if not specified
	startingModVer = "v1.0.0"
)

func init() {
	// default to the real version
	getTagsHook = getTags
}

// Execute executes the specified command.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

// wrapper for cobra, prints tag to stdout
func theCommand(args []string) error {
	tag, err := theCommandImpl(args)
	if err == nil {
		fmt.Printf("tag: %s\n", tag)
	}
	return err
}

// does the actual work
func theCommandImpl(args []string) (string, error) {
	stage := filepath.Clean(args[0])
	if len(args) == 2 {
		if !modinfo.IsValidModuleVersion(args[1]) {
			return "", fmt.Errorf("the string '%s' is not a valid module version", args[1])
		}
		startingModVer = args[1]
	}
	lmv, err := findLatestMajorVersion(stage)
	if err != nil {
		return "", fmt.Errorf("failed to find latest major version: %v", err)
	}
	mod, err := modinfo.GetModuleInfo(lmv, stage)
	if err != nil {
		return "", fmt.Errorf("failed to create module info: %v", err)
	}
	if err = writeChangelog(stage, mod); err != nil {
		return "", fmt.Errorf("failed to write changelog: %v", err)
	}
	var tag string
	if mod.BreakingChanges() {
		tag, err = forSideBySideRelease(stage, mod)
	} else {
		tag, err = forInplaceUpdate(lmv, stage, mod)
	}
	return tag, err
}

// releases the module as a new side-by-side major version
func forSideBySideRelease(stage string, mod modinfo.Provider) (string, error) {
	// update the go.mod file with the new major version
	goMod := filepath.Join(stage, "go.mod")
	file, err := os.OpenFile(goMod, os.O_RDWR, 0666)
	if err != nil {
		return "", fmt.Errorf("failed to open for read '%s': %v", goMod, err)
	}
	ver := modinfo.FindVersionSuffix(mod.DestDir())
	if err = updateGoModVer(file, ver); err != nil {
		file.Close()
		return "", fmt.Errorf("failed to update go.mod file: %v", err)
	}
	// must close file before renaming directory
	file.Close()
	// move staging to new LMV directory
	if err = os.Rename(stage, mod.DestDir()); err != nil {
		return "", fmt.Errorf("failed to rename '%s' to '%s': %v", stage, mod.DestDir(), err)
	}
	var tag string
	if tag, err = calculateModuleTag(nil, mod); err != nil {
		return "", fmt.Errorf("failed to calculate module tag: %v", err)
	}
	return tag, nil
}

// releases the module as an in-place update (minor or patch)
func forInplaceUpdate(lmv, stage string, mod modinfo.Provider) (string, error) {
	// find existing tags for this module and create a new one
	prefix, err := getTagPrefix(lmv)
	if err != nil {
		return "", fmt.Errorf("failed to get tag prefix: %v", err)
	}
	tags, err := getTagsHook(lmv, prefix)
	if err != nil {
		return "", fmt.Errorf("failed to retrieve tags: %v", err)
	}
	var tag string
	if tag, err = calculateModuleTag(tags, mod); err != nil {
		return "", fmt.Errorf("failed to calculate module tag: %v", err)
	}
	// move staging directory over the LMV by first deleting LMV then renaming stage
	if modinfo.HasVersionSuffix(lmv) {
		if err := os.RemoveAll(lmv); err != nil {
			return "", fmt.Errorf("failed to delete '%s': %v", lmv, err)
		}
		if err := os.Rename(stage, mod.DestDir()); err != nil {
			return "", fmt.Errorf("failed to rename '%s' toi '%s': %v", stage, lmv, err)
		}
		return tag, nil
	}
	// for v1 it's a bit more complicated since stage is a subdir of LMV.
	// first move stage to a temp dir outside of LMV, then remove LMV, then move temp to LMV
	dest := filepath.Dir(stage)
	temp := dest + "1temp"
	if err := os.Rename(stage, temp); err != nil {
		return "", fmt.Errorf("failed to rename '%s' to '%s': %v", stage, temp, err)
	}
	if err := os.RemoveAll(dest); err != nil {
		return "", fmt.Errorf("failed to delete '%s': %v", dest, err)
	}
	if err := os.Rename(temp, dest); err != nil {
		return "", fmt.Errorf("failed to rename '%s' to '%s': %v", temp, dest, err)
	}
	return tag, nil
}

// returns the absolute path to the latest major version based on the provided staging directory.
// it's assumed that the staging directory is a subdirectory of the actual package directory.
func findLatestMajorVersion(stage string) (string, error) {
	// example input:
	// ~/work/src/github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2018-03-01/redis/stage
	// finds:
	// ~/work/src/github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2018-03-01/redis
	// ~/work/src/github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2018-03-01/redis/v2
	// returns:
	// ~/work/src/github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2018-03-01/redis/v2
	parent := filepath.Dir(stage)
	dirs, err := modinfo.GetModuleSubdirs(parent)
	if err != nil {
		return "", fmt.Errorf("failed to get module subdirs '%s': %v", parent, err)
	}
	// no dirs means this is a v1 package
	if len(dirs) == 0 {
		return parent, nil
	}
	sort.Strings(dirs)
	// last one in the slice is the largest
	return filepath.Join(parent, dirs[len(dirs)-1]), nil
}

// updates the module version inside the go.mod file
func updateGoModVer(goMod io.ReadWriteSeeker, newVer string) error {
	scanner := bufio.NewScanner(goMod)
	scanner.Split(bufio.ScanLines)
	lines := []string{}
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	_, err := goMod.Seek(0, io.SeekStart)
	if err != nil {
		return fmt.Errorf("failed to seek to start: %v", err)
	}
	for _, line := range lines {
		if strings.Index(line, "module") > -1 {
			if modinfo.HasVersionSuffix(line) {
				line = strings.Replace(line, "/"+modinfo.FindVersionSuffix(line), "/"+newVer, 1)
			} else {
				line = line + "/" + newVer
			}
		}
		fmt.Fprintln(goMod, line)
	}
	return nil
}

func writeChangelog(stage string, mod modinfo.Provider) error {
	// don't write a changelog for a new module
	if mod.NewModule() {
		return nil
	}
	const changeLogName = "CHANGELOG.md"
	rpt := mod.GenerateReport()
	log, err := os.Create(filepath.Join(stage, changeLogName))
	if err != nil {
		return fmt.Errorf("failed to create %s: %v", changeLogName, err)
	}
	defer log.Close()
	if rpt.IsEmpty() {
		_, err = log.WriteString("No changes to exported content compared to the previous release.\n")
		return err
	}
	_, err = log.WriteString(rpt.ToMarkdown())
	return err
}

// returns a slice of tags for the specified repo and tag prefix
func getTags(repoPath, tagPrefix string) ([]string, error) {
	wt, err := repo.Get(repoPath)
	if err != nil {
		return nil, err
	}
	return wt.ListTags(tagPrefix + "*")
}

// returns the tag prefix for the specified package.
// assumes repo root of github.com/Azure/azure-sdk-for-go/
func getTagPrefix(pkgDir string) (string, error) {
	// e.g. /work/src/github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2018-03-01/redis/v2
	// would return services/redis/mgmt/2018-03-01/redis/v2.0.0
	repoRoot := filepath.Join("github.com", "Azure", "azure-sdk-for-go")
	i := strings.Index(pkgDir, repoRoot)
	if i < 0 {
		return "", fmt.Errorf("didn't find '%s' in '%s'", repoRoot, pkgDir)
	}
	return strings.Replace(pkgDir[i+len(repoRoot)+1:], "\\", "/", -1), nil
}

// returns the appropriate module tag based on the package version info
// tags - list of all current tags for the module
func calculateModuleTag(tags []string, mod modinfo.Provider) (string, error) {
	if mod.BreakingChanges() && !mod.VersionSuffix() {
		return "", errors.New("package has breaking changes but directory has no version suffix")
	}
	tagPrefix, err := getTagPrefix(mod.DestDir())
	if err != nil {
		return "", err
	}
	// if this has breaking changes then it's simply the prefix as a new major version
	if mod.BreakingChanges() {
		return tagPrefix + ".0.0", nil
	}
	if len(tags) == 0 {
		if mod.VersionSuffix() {
			panic("module contains a version suffix but no tags were found")
		}
		// this is the first module version
		return tagPrefix + "/" + startingModVer, nil
	}
	if !mod.VersionSuffix() {
		tagPrefix = tagPrefix + "/v1"
	}
	tag := tags[len(tags)-1]
	v := semverRegex.FindString(tag)
	if v == "" {
		return "", fmt.Errorf("didn't find semver in tag '%s'", tag)
	}
	sv, err := semver.NewVersion(v)
	if err != nil {
		return "", fmt.Errorf("failed to parse semver: %v", err)
	}
	// for non-breaking changes determine if this is a minor or patch update.
	if mod.NewExports() {
		// new exports, this is a minor update so bump minor version
		n := sv.IncMinor()
		sv = &n
	} else {
		// no new exports, this is a patch update
		n := sv.IncPatch()
		sv = &n
	}
	return strings.Replace(tag, v, "v"+sv.String(), 1), nil
}
