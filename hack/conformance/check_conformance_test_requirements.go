/*
Copyright 2019 The Kubernetes Authors.

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

// This tool is for checking conformance e2e tests follow the requirements
// which is https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/conformance-tests.md#conformance-test-requirements
package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"regexp"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
)

const (
	//e.g. framework.ConformanceIt("should provide secure master service ", func() {
	patternStartConformance = `framework.ConformanceIt\(.*, func\(\) {$`
	patternEndConformance   = `}\)$`
	patternSkip             = `framework.Skip.*\(`
)

// This function checks the requirement: it works for all providers (e.g., no SkipIfProviderIs/SkipUnlessProviderIs calls)
func checkAllProviders(e2eFile string) error {
	checkFailed := false
	inConformanceCode := false

	regStartConformance := regexp.MustCompile(patternStartConformance)
	regEndConformance := regexp.MustCompile(patternEndConformance)
	regSkip := regexp.MustCompile(patternSkip)

	fileInput, err := ioutil.ReadFile(e2eFile)
	if err != nil {
		return errors.Wrapf(err, "Failed to read file %s", e2eFile)
	}
	scanner := bufio.NewScanner(bytes.NewReader(fileInput))
	scanner.Split(bufio.ScanLines)

	for scanner.Scan() {
		line := scanner.Text()
		if regStartConformance.MatchString(line) {
			if inConformanceCode {
				return errors.Errorf("Missed the end of previous conformance test. There might be a bug in this script.")
			}
			inConformanceCode = true
		}
		if inConformanceCode {
			if regSkip.MatchString(line) {
				// To list all invalid places in a single operation of this tool, here doesn't return error and continues checking.
				fmt.Fprintf(os.Stderr, "%v: Conformance test should not call any framework.Skip*()\n", e2eFile)
				checkFailed = true
			}
			if regEndConformance.MatchString(line) {
				inConformanceCode = false
			}
		}
	}
	if inConformanceCode {
		return errors.Errorf("Missed the end of previous conformance test. There might be a bug in this script.")
	}
	if checkFailed {
		return errors.Errorf("We need to fix the above errors.")
	}
	return nil
}

func processFile(e2ePath string) error {
	regGoFile := regexp.MustCompile(`.*\.go`)

	files, err := ioutil.ReadDir(e2ePath)
	if err != nil {
		return errors.Wrapf(err, "Failed to read dir %s", e2ePath)
	}
	for _, file := range files {
		if file.IsDir() {
			continue
		}
		if !regGoFile.MatchString(file.Name()) {
			continue
		}
		e2eFile := fmt.Sprintf("%s/%s", e2ePath, file.Name())
		err = checkAllProviders(e2eFile)
		if err != nil {
			return err
		}
	}
	return nil
}

func processDir(e2ePath string) error {
	err := processFile(e2ePath)
	if err != nil {
		return err
	}

	// Search sub directories if exist
	files, err := ioutil.ReadDir(e2ePath)
	if err != nil {
		return errors.Wrapf(err, "Failed to read dir %s", e2ePath)
	}
	for _, file := range files {
		if !file.IsDir() {
			continue
		}
		err = processDir(fmt.Sprintf("%s/%s", e2ePath, file.Name()))
		if err != nil {
			return err
		}
	}
	return nil
}

func newCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "check_conformance_test_requirements [e2e-test-path]",
		Short: "Check conformance test code follows the requirements",
		Run: func(cmd *cobra.Command, args []string) {
			if len(args) != 1 {
				cmd.Help()
				os.Exit(1)
			}
			e2eRootPath := args[0]
			err := processDir(e2eRootPath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			os.Exit(0)
		},
	}
	return cmd
}

func main() {
	command := newCommand()
	if err := command.Execute(); err != nil {
		os.Exit(1)
	}
}
