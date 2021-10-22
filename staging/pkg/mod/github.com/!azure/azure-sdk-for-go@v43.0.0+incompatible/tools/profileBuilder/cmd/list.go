// +build go1.9

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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/Azure/azure-sdk-for-go/tools/internal/dirs"
	"github.com/Azure/azure-sdk-for-go/tools/internal/modinfo"
	"github.com/Azure/azure-sdk-for-go/tools/profileBuilder/model"
	"github.com/spf13/cobra"
)

const (
	inputLongName    = "input"
	inputShortName   = "i"
	inputDescription = "Specify the input JSON file to read for the list of packages."
)

// listCmd represents the list command
var listCmd = &cobra.Command{
	Use:   "list",
	Short: "Creates a profile from a set of packages.",
	Long: `Reads a list of packages from stdin, where each line is treated as a Go package
identifier. These packages are then used to create a profile.

Often, the easiest way of invoking this command will be using a pipe operator
to specify the packages to include.

Example:
$> ../model/testdata/smallProfile.txt > profileBuilder list --name small_profile
`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		logWriter := ioutil.Discard
		if verboseFlag {
			logWriter = os.Stdout
		}

		outputLog := log.New(logWriter, "[STATUS] ", 0)
		errLog := log.New(os.Stderr, "[ERROR] ", 0)

		if !filepath.IsAbs(outputRootDir) {
			abs, err := filepath.Abs(outputRootDir)
			if err != nil {
				errLog.Fatalf("failed to convert to absolute path: %v", err)
			}
			outputRootDir = abs
		}

		inputFile, err := cmd.Flags().GetString(inputLongName)
		if err != nil {
			errLog.Fatalf("failed to get %s: %v", inputLongName, err)
		}

		data, err := ioutil.ReadFile(inputFile)
		if err != nil {
			errLog.Fatalf("failed to read list: %v", err)
		}

		var listDef model.ListDefinition
		err = json.Unmarshal(data, &listDef)
		if err != nil {
			errLog.Fatalf("failed to unmarshal JSON: %v", err)
		}

		if modulesFlag {
			// when generating in module-aware mode a few extra things need to happen
			// 1. find the latest module version for the profile we're working on
			// 2. update the list of includes package to their latest module versions
			// 3. if any includes were updated generate a new major version for this profile
			modver, err := getLatestModVer(outputRootDir)
			if err != nil {
				errLog.Fatalf("failed to get module dir: %v", err)
			}
			updated, err := updateModuleVersions(&listDef)
			if err != nil {
				errLog.Fatalf("failed to update module versions: %v", err)
			}
			if updated {
				// at least one include was updated, write out updated list definition
				data, err = json.MarshalIndent(listDef, "", "  ")
				if err != nil {
					errLog.Fatalf("failed to marshal updated list: %v", err)
				}
				data = append(data, '\n')
				err = ioutil.WriteFile(inputFile, data, 0666)
				if err != nil {
					errLog.Fatalf("failed to write updated list: %v", err)
				}
				// increment profile module major version and generate new go.mod file
				modver = modinfo.IncrementModuleVersion(modver)
				outputRootDir = filepath.Join(outputRootDir, modver)
				err = generateGoMod(outputRootDir)
				if err != nil {
					errLog.Fatalf("failed to generate go.mod: %v", err)
				}
			} else if modver != "" {
				outputRootDir = filepath.Join(outputRootDir, modver)
			}
		}
		fmt.Printf("Executes profileBuilder in %s\n", outputRootDir)
		outputLog.Printf("Output-Location set to: %s", outputRootDir)
		if clearOutputFlag {
			if err := dirs.DeleteChildDirs(outputRootDir); err != nil {
				errLog.Fatalf("Unable to clear output-folder: %v", err)
			}
		}
		// use recursive build to include the *api packages
		model.BuildProfile(listDef, profileName, outputRootDir, outputLog, errLog, true, modulesFlag)
	},
}

func init() {
	rootCmd.AddCommand(listCmd)
	listCmd.Flags().StringP(inputLongName, inputShortName, "", inputDescription)
	listCmd.MarkFlagRequired(inputLongName)
}

// for each included package check if there is a newer module major version then the one specified,
// and if there is update the include accordinly.  returns true if any packages were updated.
func updateModuleVersions(listDef *model.ListDefinition) (bool, error) {
	dirty := false
	for i, entry := range listDef.Include {
		// "../../services/storage/mgmt/2016-01-01/storage"
		// "../../services/network/mgmt/2015-06-15/network/v2"
		target := entry
		if modinfo.HasVersionSuffix(target) {
			target = filepath.Dir(target)
			target = strings.Replace(target, "\\", "/", -1)
		}
		modDirs, err := modinfo.GetModuleSubdirs(target)
		if err != nil {
			return false, err
		}
		if len(modDirs) == 0 {
			continue
		}
		latest := target + "/" + modDirs[len(modDirs)-1]
		if latest == entry {
			// already using latest major version
			continue
		}
		listDef.Include[i] = latest
		dirty = true
	}
	return dirty, nil
}

// returns the last major module version for the specified profile directory (e.g. "v2" etc).
// if there are no major versions the return value is the empty string.
func getLatestModVer(profileDir string) (string, error) {
	subdirs, err := modinfo.GetModuleSubdirs(profileDir)
	if err != nil {
		return "", err
	}
	modDir := ""
	if len(subdirs) > 0 {
		modDir = subdirs[len(subdirs)-1]
	}
	return modDir, nil
}

// generate a go.mod file in the specified module major version directory (e.g. "profiles/foo/v2")
// the provided module directory is used to calculate the module name.
func generateGoMod(modDir string) error {
	const gomodFormat = "module %s\n\ngo 1.12\n"
	err := os.Mkdir(modDir, os.ModeDir|0644)
	if err != nil {
		return err
	}
	gomod, err := os.Create(filepath.Join(modDir, "go.mod"))
	if err != nil {
		return err
	}
	defer gomod.Close()
	mod, err := modinfo.CreateModuleNameFromPath(modDir)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(gomod, gomodFormat, mod)
	return err
}
