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
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
)

var exceptFileFlag string

var rootCmd = &cobra.Command{
	Use:   "pkgchk <dir>",
	Short: "Performs package validation tasks against all packages found under the specified directory.",
	Long: `This tool will perform various package validation checks against all of the packages
found under the specified directory.  Failures can be baselined and thus ignored by
copying the failure text verbatim, pasting it into a text file then specifying that
file via the optional exceptions flag.
`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		return theCommand(args)
	},
}

func init() {
	rootCmd.PersistentFlags().StringVarP(&exceptFileFlag, "exceptions", "e", "", "text file containing the list of exceptions")
}

// Execute executes the specified command.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func theCommand(args []string) error {
	rootDir, err := filepath.Abs(args[0])
	if err != nil {
		return errors.Wrap(err, "failed to get absolute path")
	}
	pkgs, err := getPkgs(rootDir)
	if err != nil {
		return errors.Wrap(err, "failed to get packages")
	}
	exceptions, err := loadExceptions(exceptFileFlag)
	if err != nil {
		return errors.Wrap(err, "failed to load exceptions")
	}
	verifiers := getVerifiers()
	count := 0
	for _, pkg := range pkgs {
		for _, v := range verifiers {
			if err = v(pkg); err != nil && !contains(exceptions, err.Error()) {
				fmt.Fprintln(os.Stderr, err)
				count++
			}
		}
	}
	if count > 0 {
		return fmt.Errorf("found %d errors", count)
	}
	return nil
}

func contains(items []string, item string) bool {
	if items == nil {
		return false
	}
	for _, i := range items {
		if i == item {
			return true
		}
	}
	return false
}

func loadExceptions(exceptFile string) ([]string, error) {
	if exceptFile == "" {
		return nil, nil
	}
	f, err := os.Open(exceptFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	exceps := []string{}

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		exceps = append(exceps, scanner.Text())
	}
	if err = scanner.Err(); err != nil {
		return nil, err
	}

	return exceps, nil
}

type pkg struct {
	// the directory where the package resides relative to the root dir
	d string

	// the AST of the package
	p *ast.Package
}

// returns true if the package directory corresponds to an ARM package
func (p pkg) isARMPkg() bool {
	return strings.Index(p.d, "/mgmt/") > -1
}

// walks the directory hierarchy from the specified root returning a slice of all the packages found
func getPkgs(rootDir string) ([]pkg, error) {
	pkgs := make([]pkg, 0)
	err := filepath.Walk(rootDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			// check if leaf dir
			fi, err := ioutil.ReadDir(path)
			if err != nil {
				return err
			}
			hasSubDirs := false
			interfacesDir := false
			for _, f := range fi {
				if f.IsDir() {
					hasSubDirs = true
					break
				}
				if f.Name() == "interfaces.go" {
					interfacesDir = true
				}
			}
			if !hasSubDirs {
				fs := token.NewFileSet()
				// with interfaces codegen the majority of leaf directories are now the
				// *api packages. when this is the case parse from the parent directory.
				if interfacesDir {
					path = filepath.Dir(path)
				}
				packages, err := parser.ParseDir(fs, path, func(fi os.FileInfo) bool {
					return fi.Name() != "interfaces.go"
				}, parser.PackageClauseOnly)
				if err != nil {
					return err
				}
				if len(packages) < 1 {
					return errors.New("didn't find any packages which is unexpected")
				}
				if len(packages) > 1 {
					return errors.New("found more than one package which is unexpected")
				}
				var p *ast.Package
				for _, pkgs := range packages {
					p = pkgs
				}
				// normalize directory separator to '/' character
				pkgs = append(pkgs, pkg{
					d: strings.Replace(path[len(rootDir):], "\\", "/", -1),
					p: p,
				})
			}
		}
		return nil
	})
	return pkgs, err
}

type verifier func(p pkg) error

// returns a list of verifiers to execute
func getVerifiers() []verifier {
	return []verifier{
		verifyPkgMatchesDir,
		verifyLowerCase,
		verifyDirectoryStructure,
	}
}

// ensures that the leaf directory name matches the package name
// new to modules: if the last leaf is version suffix, find its parent as leaf folder
func verifyPkgMatchesDir(p pkg) error {
	leaf := findPackageFolderInPath(p.d)
	if !strings.EqualFold(leaf, p.p.Name) {
		return fmt.Errorf("leaf directory of '%s' doesn't match package name '%s'", p.d, p.p.Name)
	}
	return nil
}

func findPackageFolderInPath(path string) string {
	regex := regexp.MustCompile(`/v\d+$`)
	if regex.MatchString(path) {
		// folder path ends with version suffix
		path = path[:strings.LastIndex(path, "/")]
	}
	result := path[strings.LastIndex(path, "/")+1:]
	return result
}

// ensures that there are no upper-case letters in a package's directory
func verifyLowerCase(p pkg) error {
	// walk the package directory looking for upper-case characters
	for _, r := range p.d {
		if r == '/' {
			continue
		}
		if unicode.IsUpper(r) {
			return fmt.Errorf("found upper-case character in directory '%s'", p.d)
		}
	}
	return nil
}

// ensures that the package's directory hierarchy is properly formed
func verifyDirectoryStructure(p pkg) error {
	// for ARM the package directory structure is highly deterministic:
	// /redis/mgmt/2015-08-01/redis
	// /resources/mgmt/2017-06-01-preview/policy
	// /preview/signalr/mgmt/2018-03-01-preview/signalr
	// /preview/security/mgmt/v2.0/security (version scheme for composite packages)
	// /network/mgmt/2019-10-01/network/v2 (new with modules)
	if !p.isARMPkg() {
		return nil
	}
	regexStr := strings.Join([]string{
		`^(?:/preview)?`,
		`[a-z0-9\-]+`,
		`mgmt`,
		`((?:\d{4}-\d{2}-\d{2}(?:-preview)?)|(?:v\d{1,2}\.\d{1,2}))`,
		`[a-z0-9]+`,
	}, "/")
	regexStr = regexStr + `(/v\d+)?$`
	regex := regexp.MustCompile(regexStr)
	if !regex.MatchString(p.d) {
		return fmt.Errorf("bad directory structure '%s'", p.d)
	}
	return nil
}
