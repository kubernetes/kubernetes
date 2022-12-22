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

// verifies the output of a command with the expected output
// There are 2 states : start and Parsing.
// start ---Run:--> Parsing (Capture Run)
// Parsing ---Run:---> Parsing (Reinitialize Run)
// Parsing ---Metadata:---> Parse Metadata
// Parsing ---Expected output:---> start (Capture verify and goto start)
/*
The parsing is done using an Finite State Machine(FSM). The FSM has 2 states Start and Parsing. The initial state is Start. The parsing logic is summarized in the following table.

| State | Input | Action | Output State
|------ |-------|--------|-------------|
| Start | Run Block | Record the command| Parsing|
| Parsing | Run Block | Record the command| Parsing|
| Parsing | Meta Block | Record the metadata| Parsing|
| Parsing | Verify Block | Record the output | Start|

An alternative is to use a markdown parser but that would help us only in the Action column and it would introduce external dependencies.
*/

package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io/fs"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/google/go-cmp/cmp"
	"k8s.io/klog/v2"
)

// FSM State of the parsing.
type state uint32

const (
	// Initial state waiting for Run: cmd
	start state = iota
	// Middle of parsing Run/Verify
	parsing
)

const (
	runCmd = iota
	metaCmd
	verifyCmd
)

const (
	RunTrigger    = "Run:"
	MetaTrigger   = "Metadata:"
	OutputTrigger = "Expected output:"
	StartConsole  = "```console"
	EndConsole    = "```"
)

const newline = "\n"

var failFast = flag.Bool("failfast", false, "specify failfast to fail immediately")

// state-command combination
type commandState struct {
	state   state
	command uint32
}

type metaData struct {
	headerRegex []string
}

// Run-verify combo
type runVerify struct {
	run    string
	verify string
	dir    string
	meta   metaData
}

// Common regexes (Currently we support klog and json)
// [1] klog regex is based on the header formatted by klog.go
// Sample: I0412 01:55:01.765514  728443 logger.go:76] Log using Infof, key: value
// We name the first named group as klog_header. The second unnamed group (.*) is used as-is
// [2] json header
// {"ts":1648546681782.9392,"caller":"cmd/logger.go:77","msg"... => {"msg"...
var Regex_header = map[string]string{
	"ignore_klog_header": "(?m)(?P<klog_header>^[IWEF][0-9]{4}\\s[0-9]{2}:[0-9]{2}:[0-9]{2}\\.[0-9]{6}\\s*[0-9]*\\s.*\\.go:[0-9]{2}])(.*)",
	"ignore_json_header": "(?m)(?P<json_header>\"ts\":\\d*.\\d*\\,\"caller\":\"[0-9A-Za-z/.]*:\\d*\",?)(.*)"}

// This function processes the headers in a log file, typically timestamps and other
// location specific information.
func processHeaders(input string, ignoreHeader []string) (bool, string) {
	output := ""
	for _, s := range ignoreHeader {
		//re init the output
		output = ""

		// compile the regex
		r := regexp.MustCompile(s)

		matched := r.FindAllStringSubmatchIndex(input, -1)
		pos := 0

		//TODO: Optimize & make this inplace. Use strings package and replace.
		for _, slice := range matched {
			// make sure you include text not captured by the regex.
			if pos < slice[0] {
				output = output + input[pos:slice[0]]
				pos = slice[0]
			}
			output = output + r.SubexpNames()[1] + input[slice[4]:slice[5]]
			pos = slice[5]
		}

		// Include trailing text
		if pos < len(input) {
			output = output + input[pos:]
		}

		// update the input
		input = output
	}

	return true, input
}

// Process the logs and then compare
func (rv *runVerify) compare(output string) error {

	var ignoreHeader = []string{}

	// User-specified regexes
	for _, s := range rv.meta.headerRegex {
		// Map keywords to supported regexes
		if r, ok := Regex_header[s]; ok {
			s = r
		}
		ignoreHeader = append(ignoreHeader, s)
	}
	klog.V(2).InfoS("Ignore Headers", "regexes", ignoreHeader)

	// process headers of output log and expected log
	status, processedOutput := processHeaders(output, ignoreHeader)
	status2 := false
	processedExpected := ""
	if status {
		status2, processedExpected = processHeaders(rv.verify, ignoreHeader)
		if status2 {
			if processedOutput != processedExpected {
				// Mismatch after processing headers
				err := fmt.Errorf("Mismatch in Run command : %v\n", rv.run)
				// Print the diff to stdout
				fmt.Printf("Compare Failure! Error : %v Diff : %v\n", err,
					cmp.Diff(processedExpected, processedOutput))
				return err
			}
		}
	}

	if !status || !status2 {
		klog.V(2).InfoS("Cannot process headers, so falling back to plain text processing")

		// Fallback to comparing non-processed output
		if output != rv.verify {
			err := fmt.Errorf("Mismatch in Run command : %v\n", rv.run)
			// Print the diff to stdout
			fmt.Printf("Compare Failure! Error : %v Diff : %v\n", err,
				cmp.Diff(rv.verify, output))
			return err
		}
	}
	return nil
}

func (rv *runVerify) validate(mdfile string) error {

	fcmd := strings.Fields(rv.run)
	binary := fcmd[0]
	args := fcmd[1:]
	cmd := exec.Command(binary, args...)
	cmd.Dir = rv.dir
	ex, _ := os.Executable()
	klog.V(2).InfoS("==================New Run====================", "markdown-file", mdfile)
	klog.V(2).InfoS("Binary related information", "binary", binary, "arguments",
		args, "directory", cmd.Dir, "run-verify directory", rv.dir, "exec-name", filepath.Base(ex))
	// Running go run . in the checkoutput directory causes a hang.
	if filepath.Base(ex) == "checkoutput" {
		klog.V(2).InfoS("Skip as we will recursively loop in checkoutput tool in validate")
		return fmt.Errorf("Cannot call checkoutput tool in validate\n")
	}
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("Validation Failure: failed to execute command %v \n error: \n %v \ncommand-output: %s", rv.run, fmt.Sprint(err), output)
	}

	// TODO: Currently we capture the output of just the stdout. Logs extensively write to
	// stderr, we need to come up with the right logic for validation
	err = rv.compare(string(output))
	if err == nil {
		klog.V(2).InfoS("Validation Success", "Run Command", rv.run, "Verified text", rv.verify)
		return nil
	}

	klog.V(2).InfoS("Failure", "markdown-file", mdfile)
	return err
}

type transitionFunc func(state *state, rv *runVerify, scanner *bufio.Scanner, mdfile string) error

func parseRunCmd(state *state, rv *runVerify, scanner *bufio.Scanner, mdfile string) error {
	scanner.Scan()
	if scanner.Text() == "```console" {
		//scan the command
		scanner.Scan()
		run := scanner.Text()
		scanner.Scan()
		if scanner.Text() == "```" {
			rv.run = run
			// chuck any previous metadata
			rv.meta.headerRegex = nil
			// chuck any previous verify
			rv.verify = ""
			*state = parsing
			// successful Run command parsing.
			return nil
		}
	} else {
		return fmt.Errorf("expected ```console but got %v", scanner.Text())
	}
	return fmt.Errorf("failed to parse Run command")
}

func parseMetaCmd(state *state, rv *runVerify, scanner *bufio.Scanner, mdfile string) error {
	scanner.Scan()
	if scanner.Text() == "```console" {
		for scanner.Scan() {
			//scan the command
			meta := scanner.Text()
			if meta == "```" {
				return nil
			}
			rv.meta.headerRegex = append(rv.meta.headerRegex, meta)
		}
	} else {
		return fmt.Errorf("expected ```console but got %v", scanner.Text())
	}
	return fmt.Errorf("failed to parse Meta command")
}

func parseVerifyCmd(state *state, rv *runVerify, scanner *bufio.Scanner, mdfile string) error {
	if len(rv.run) == 0 {
		return fmt.Errorf("cannot have verification without a corresponding Run command")
	}
	var sb strings.Builder
	scanner.Scan()
	if scanner.Text() == "```console" {
		for scanner.Scan() {
			if scanner.Text() == "```" {
				rv.verify = sb.String()
				*state = start
				return rv.validate(mdfile)
			}
			sb.WriteString(scanner.Text())
			// Is this right ? For windows \r\n is also possible ?
			sb.WriteString(newline)
		}
	} else {
		return fmt.Errorf("verify: Expected ```console but got %v", scanner.Text())
	}
	return fmt.Errorf("failed to parse Expected output")
}

var stateTransitionTable = map[commandState]transitionFunc{
	{start, runCmd}:      parseRunCmd,
	{parsing, runCmd}:    parseRunCmd,
	{parsing, metaCmd}:   parseMetaCmd,
	{parsing, verifyCmd}: parseVerifyCmd,
}

func parseMD(ctx context.Context, mdfile string, dir string) error {
	logger := klog.FromContext(ctx)

	f, err := os.Open(mdfile)
	if err != nil {
		return fmt.Errorf("Cannot open input markdown file: %v", err)
	}
	defer f.Close()

	re_run := regexp.MustCompile("^Run:")
	re_meta := regexp.MustCompile("^Metadata:")
	re_verify := regexp.MustCompile("^Expected output:")

	state := start
	var rv runVerify
	rv.dir = dir

	logger.V(2).Info("Parsing Markdown File", "markdown file", mdfile, "dir", dir)

	scanner := bufio.NewScanner(f)
	var cmd uint32
	var match bool
	for scanner.Scan() {
		match = false
		line := scanner.Text()
		if re_run.MatchString(line) {
			match = true
			cmd = runCmd
		} else if re_meta.MatchString(line) {
			match = true
			cmd = metaCmd
		} else if re_verify.MatchString(line) {
			match = true
			cmd = verifyCmd
		}
		if match {
			tupple := commandState{state, cmd}
			if f := stateTransitionTable[tupple]; f != nil {
				err := f(&state, &rv, scanner, mdfile)
				if err != nil {
					if *failFast {
						return err
					} else {
						logger.Error(err, "State transition Failure", "mdfile", mdfile, "dir", dir)
					}
				}
			}
		}
	}
	if state != start {
		return fmt.Errorf("Parse error of file %v", mdfile)
	}
	return nil
}

func getMainDirList(dir string, mainList []string) ([]string, error) {

	re_main := regexp.MustCompile("(?m)^package main")
	err := filepath.Walk(dir, func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			klog.ErrorS(err, "Cannot Walk")
			return err
		}

		if info.IsDir() {
			name := info.Name()
			// Ignore hidden directories (.git, .cache, etc)
			if (len(name) > 1 && (name[0] == '.' || name[0] == '_')) || name == "testdata" {
				klog.V(2).InfoS("Skipping directory", "path", path)
				return filepath.SkipDir
			}
		}

		go_file := strings.HasSuffix(path, ".go")
		if go_file {
			data, err := ioutil.ReadFile(path)
			if err != nil {
				klog.ErrorS(err, "Cannot load go file")
				return err
			}
			if re_main.MatchString(string(data)) {
				klog.V(2).InfoS("Found main module!", "path", path)
				mainList = append(mainList, filepath.Dir(path))
			}
		}

		return nil
	})
	if err != nil {
		return mainList, err
	}
	return mainList, nil
}

func processDir(roots []string) error {
	var mainList []string
	var err error

	//Get a list of directories that have the main package
	for _, root := range roots {
		klog.V(2).InfoS("Processing root directory", "dir", root)

		mainList, err = getMainDirList(root, mainList)
		if err != nil {
			return err
		}
		klog.V(2).InfoS("List with main modules/packages", "mainList", mainList)
	}

	// Filter based on the presence of the md file.
	for _, dir := range mainList {
		err := filepath.Walk(dir, func(path string, info fs.FileInfo, err error) error {
			// skip directories as we are interested in markdown files
			if info.IsDir() && info.Name() != filepath.Base(dir) {
				return filepath.SkipDir
			}

			md := strings.HasSuffix(path, ".md")
			if md {
				err := parseMD(context.Background(), path, dir)
				if *failFast {
					return err
				}
			}
			return nil
		})

		if err != nil {
			return fmt.Errorf("error walking the path for markdown file %v\n", err)
		}
	}

	return nil
}

func main() {
	klog.InitFlags(nil)
	flag.Parse()
	args := flag.Args()
	if len(args) == 0 {
		args = append(args, ".")
	}
	klog.V(2).InfoS("verbosity level", "v", flag.CommandLine.Lookup("v").Value)
	err := processDir(args)
	if err != nil {
		fmt.Println(err)
	}
}
