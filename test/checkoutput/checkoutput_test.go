package main

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
)

const pathDivider = "/"

type EXEC_TYPE int

const (
	CMD_EXEC EXEC_TYPE = iota + 1
	MDFILE_EXEC
)

type testCase struct {
	// name of the testCase
	name string

	// path to the program to be tested
	pathProgram string

	// execution from cmd, mdfile etc
	execType EXEC_TYPE

	//input to EXEC_TYPE, either a cmd or a mdfile
	input string

	// expected success or failure
	result bool

	// mutate flag
	mutate bool
}

func getFullPath(suffix string, tcase *testCase) string {
	return "../../" + tcase.pathProgram + "/" + suffix
}

func writeWithNewline(sb *strings.Builder, str string) {
	sb.WriteString(str)
	sb.WriteString("\n")
}

func writeWithConsole(sb *strings.Builder, str string) {
	writeWithNewline(sb, StartConsole)
	writeWithNewline(sb, str)
	writeWithNewline(sb, EndConsole)
}

func constructMdContent(cmd string, defaultMeta bool, customMeta []string, output string) string {
	var sb strings.Builder
	// Write the command
	writeWithNewline(&sb, RunTrigger)
	writeWithConsole(&sb, cmd)

	// Write the metadata
	if defaultMeta || len(customMeta) > 0 {
		writeWithNewline(&sb, MetaTrigger)
		writeWithNewline(&sb, StartConsole)

		if defaultMeta {
			for k, _ := range Regex_header {
				writeWithNewline(&sb, k)
			}
		}
		for _, meta := range customMeta {
			writeWithNewline(&sb, meta)
		}
		writeWithNewline(&sb, EndConsole)
	}

	// Write the output
	writeWithNewline(&sb, OutputTrigger)
	writeWithNewline(&sb, StartConsole)
	// Do not write a newline
	sb.WriteString(output)
	writeWithNewline(&sb, EndConsole)

	return sb.String()
}

func mutateOutput(input string) (string, error) {
	return string(input) + "Random Text to make the test to fail!\n", nil

}

func createAndWriteMdFile(mdFile string, content string, tcase *testCase) error {
	absMdFilePath, _ := filepath.Abs(getFullPath(mdFile, tcase))
	f, err := os.Create(absMdFilePath)
	if err != nil {
		return fmt.Errorf("could not create file %v", err)
	}
	_, err = f.WriteString(content)
	if err != nil {
		return fmt.Errorf("could not write to file %v", err)
	}

	return nil
}

func constructAndWriteContent(tcase *testCase, mutate bool, defaultMeta bool, mdFile string, exp_output string) error {
	var err error
	if mutate {
		exp_output, err = mutateOutput(exp_output)
		if err != nil {
			return fmt.Errorf("unable to mutate input")
		}
	}
	// Generate content
	content := constructMdContent(tcase.input, defaultMeta, nil, exp_output)
	// Write content to files
	err = createAndWriteMdFile(mdFile, content, tcase)
	if err != nil {
		return err
	}
	return nil
}

// Generate a new mdFile from configuration
func testGenMdFile(ctx context.Context, tcase *testCase) (string, string, error) {
	logger := klog.FromContext(ctx)
	var genMdFile, genMutatedMdFile string
	//generate the mdFile name
	rand.Seed(time.Now().UnixNano())
	min := 1000
	max := 9000
	if tcase.execType == CMD_EXEC {
		words := strings.Fields(tcase.input)
		randStr := strconv.Itoa(rand.Intn(max-min+1) + min)
		genMdFile = tcase.name + "_" + randStr + "_" + words[0] + ".md"
		if tcase.mutate {
			genMutatedMdFile = tcase.name + "_" + "mutated" + "_" + randStr + "_" + words[0] + ".md"
		}
	} else {
		return "", "", fmt.Errorf("invalid exec type!")
	}

	fcmd := strings.Fields(tcase.input)
	binary := fcmd[0]
	args := fcmd[1:]
	cmd := exec.Command(binary, args...)
	cmd.Dir = getFullPath("", tcase)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", "", fmt.Errorf("error in execing %v while generating %v, output %v, error %v",
			tcase.input, genMdFile, output, err)
	}

	// construct normal content
	err = constructAndWriteContent(tcase, false, true, genMdFile, string(output))
	if err != nil {
		return "", "", err
	}
	// construct mutated content
	if tcase.mutate {
		err = constructAndWriteContent(tcase, true, true, genMutatedMdFile, string(output))
		if err != nil {
			return "", "", err
		}
	}

	logger.Info("Generated md file name", "name", getFullPath(genMdFile, tcase),
		"mutatedName", getFullPath(genMutatedMdFile, tcase))

	return getFullPath(genMdFile, tcase), getFullPath(genMutatedMdFile, tcase), nil
}

// Sets up the test directory as per the testCase and returns
// either error or the mdFile having the test scenario
func testSetup(ctx context.Context, tcase *testCase) (string, string, error) {
	var genMdFile, genMutatedMdFile string
	logger := klog.FromContext(ctx)

	if tcase.pathProgram == "" {
		return "", "", fmt.Errorf("path to program is empty")
	}

	//sanitize the testCase
	_, err := os.Stat(getFullPath("", tcase))
	if err != nil {
		return "", "", fmt.Errorf("path to test is not found : %v", err)
	}

	if tcase.execType == MDFILE_EXEC {
		mdFile := getFullPath(tcase.input, tcase)
		_, err = os.Stat(mdFile)
		if err != nil {
			return "", "", fmt.Errorf("mdFile not found : %v", err)
		}

		if tcase.mutate {
			return "", "", fmt.Errorf("MDFILE EXEC_TYPE does not support mutate option")
		}

		return mdFile, "", nil
	} else if tcase.execType == CMD_EXEC {
		if tcase.input == "" {
			return "", "", fmt.Errorf("a CMD_EXEC type does not have a command!")
		}

		//generate the mdfile(s)
		genMdFile, genMutatedMdFile, err = testGenMdFile(ctx, tcase)
		if err != nil {
			return "", "", fmt.Errorf("problem with generating mdFile : %v", err)
		}

		logger.V(2).Info("test setup is complete", "mdfile", genMdFile,
			"mutatedfile", genMutatedMdFile)
		return genMdFile, genMutatedMdFile, nil
	}
	return "", "", fmt.Errorf("invalid exec type!")

}

func testCleanup(ctx context.Context, mdFile string, mutatedMdFile string, tcase testCase) {
	logger := klog.FromContext(ctx)
	logger.Info("Performing cleanup", "mdFile", mdFile,
		"mutatedMdFile", mutatedMdFile, "tcase", tcase)
	//Perform delete only for generated files
	if tcase.execType == CMD_EXEC {
		logger.Info("Removing mdfile", "mdfile", mdFile)
		os.Remove(mdFile)
		if tcase.mutate {
			os.Remove(mutatedMdFile)
		}
	}
}

func TestCheckoutput(t *testing.T) {
	var tests = []testCase{
		{"component-base-log-example", "staging/src/k8s.io/component-base/logs/example",
			CMD_EXEC, "go run .", true, false},
		{"component-base-log-example-with-mutate", "staging/src/k8s.io/component-base/logs/example",
			CMD_EXEC, "go run .", true, true},
		{"component-base-log-example-mdfile", "staging/src/k8s.io/component-base/logs/example",
			MDFILE_EXEC, "sample_input.md", true, false},
		{"component-base-log-example-mdfile-fail", "staging/src/k8s.io/component-base/logs/example",
			MDFILE_EXEC, "sample_input_failed.md", false, false},
	}

	for _, test := range tests {
		logger, ctx := ktesting.NewTestContext(t)
		logger = klog.LoggerWithValues(logger, "testName", test.name)
		ctx = klog.NewContext(ctx, logger)

		//setup the test
		mdFile, mutatedMdFile, err := testSetup(ctx, &test)
		if err != nil {
			t.Errorf("test %s failed during setup : error %v: input : %v", test.name, err, test)
			continue
		}
		// Perform cleanup
		defer testCleanup(ctx, mdFile, mutatedMdFile, test)

		// run the test
		// Enable failfast so that we can retrieve the error immediately
		*failFast = true
		err = parseMD(ctx, mdFile, getFullPath("", &test))
		if err != nil && test.result {
			t.Errorf("test %s failed | error : %v | input : %v", test.name, err, test)
			continue
		}
		if test.mutate {
			err = parseMD(ctx, mutatedMdFile, getFullPath("", &test))
			if err == nil {
				t.Errorf("test %s failed : mutated output should fail the test : input : %v", test.name, test)
				continue
			}
		}

		// Perform the cleanup
		//testCleanup(ctx, mdFile, mutatedMdFile, &test)
		logger.Info("--------test is successful--------")
	}
}
