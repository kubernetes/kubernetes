package dockerfile

import (
	"io/ioutil"
	"strings"
	"testing"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/builder/dockerfile/parser"
	"github.com/docker/docker/builder/remotecontext"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/reexec"
)

type dispatchTestCase struct {
	name, dockerfile, expectedError string
	files                           map[string]string
}

func init() {
	reexec.Init()
}

func initDispatchTestCases() []dispatchTestCase {
	dispatchTestCases := []dispatchTestCase{{
		name: "copyEmptyWhitespace",
		dockerfile: `COPY
	quux \
      bar`,
		expectedError: "COPY requires at least two arguments",
	},
		{
			name:          "ONBUILD forbidden FROM",
			dockerfile:    "ONBUILD FROM scratch",
			expectedError: "FROM isn't allowed as an ONBUILD trigger",
			files:         nil,
		},
		{
			name:          "ONBUILD forbidden MAINTAINER",
			dockerfile:    "ONBUILD MAINTAINER docker.io",
			expectedError: "MAINTAINER isn't allowed as an ONBUILD trigger",
			files:         nil,
		},
		{
			name:          "ARG two arguments",
			dockerfile:    "ARG foo bar",
			expectedError: "ARG requires exactly one argument",
			files:         nil,
		},
		{
			name:          "MAINTAINER unknown flag",
			dockerfile:    "MAINTAINER --boo joe@example.com",
			expectedError: "Unknown flag: boo",
			files:         nil,
		},
		{
			name:          "ADD multiple files to file",
			dockerfile:    "ADD file1.txt file2.txt test",
			expectedError: "When using ADD with more than one source file, the destination must be a directory and end with a /",
			files:         map[string]string{"file1.txt": "test1", "file2.txt": "test2"},
		},
		{
			name:          "JSON ADD multiple files to file",
			dockerfile:    `ADD ["file1.txt", "file2.txt", "test"]`,
			expectedError: "When using ADD with more than one source file, the destination must be a directory and end with a /",
			files:         map[string]string{"file1.txt": "test1", "file2.txt": "test2"},
		},
		{
			name:          "Wildcard ADD multiple files to file",
			dockerfile:    "ADD file*.txt test",
			expectedError: "When using ADD with more than one source file, the destination must be a directory and end with a /",
			files:         map[string]string{"file1.txt": "test1", "file2.txt": "test2"},
		},
		{
			name:          "Wildcard JSON ADD multiple files to file",
			dockerfile:    `ADD ["file*.txt", "test"]`,
			expectedError: "When using ADD with more than one source file, the destination must be a directory and end with a /",
			files:         map[string]string{"file1.txt": "test1", "file2.txt": "test2"},
		},
		{
			name:          "COPY multiple files to file",
			dockerfile:    "COPY file1.txt file2.txt test",
			expectedError: "When using COPY with more than one source file, the destination must be a directory and end with a /",
			files:         map[string]string{"file1.txt": "test1", "file2.txt": "test2"},
		},
		{
			name:          "JSON COPY multiple files to file",
			dockerfile:    `COPY ["file1.txt", "file2.txt", "test"]`,
			expectedError: "When using COPY with more than one source file, the destination must be a directory and end with a /",
			files:         map[string]string{"file1.txt": "test1", "file2.txt": "test2"},
		},
		{
			name:          "ADD multiple files to file with whitespace",
			dockerfile:    `ADD [ "test file1.txt", "test file2.txt", "test" ]`,
			expectedError: "When using ADD with more than one source file, the destination must be a directory and end with a /",
			files:         map[string]string{"test file1.txt": "test1", "test file2.txt": "test2"},
		},
		{
			name:          "COPY multiple files to file with whitespace",
			dockerfile:    `COPY [ "test file1.txt", "test file2.txt", "test" ]`,
			expectedError: "When using COPY with more than one source file, the destination must be a directory and end with a /",
			files:         map[string]string{"test file1.txt": "test1", "test file2.txt": "test2"},
		},
		{
			name:          "COPY wildcard no files",
			dockerfile:    `COPY file*.txt /tmp/`,
			expectedError: "COPY failed: no source files were specified",
			files:         nil,
		},
		{
			name:          "COPY url",
			dockerfile:    `COPY https://index.docker.io/robots.txt /`,
			expectedError: "source can't be a URL for COPY",
			files:         nil,
		},
		{
			name:          "Chaining ONBUILD",
			dockerfile:    `ONBUILD ONBUILD RUN touch foobar`,
			expectedError: "Chaining ONBUILD via `ONBUILD ONBUILD` isn't allowed",
			files:         nil,
		},
		{
			name:          "Invalid instruction",
			dockerfile:    `foo bar`,
			expectedError: "unknown instruction: FOO",
			files:         nil,
		}}

	return dispatchTestCases
}

func TestDispatch(t *testing.T) {
	testCases := initDispatchTestCases()

	for _, testCase := range testCases {
		executeTestCase(t, testCase)
	}
}

func executeTestCase(t *testing.T, testCase dispatchTestCase) {
	contextDir, cleanup := createTestTempDir(t, "", "builder-dockerfile-test")
	defer cleanup()

	for filename, content := range testCase.files {
		createTestTempFile(t, contextDir, filename, content, 0777)
	}

	tarStream, err := archive.Tar(contextDir, archive.Uncompressed)

	if err != nil {
		t.Fatalf("Error when creating tar stream: %s", err)
	}

	defer func() {
		if err = tarStream.Close(); err != nil {
			t.Fatalf("Error when closing tar stream: %s", err)
		}
	}()

	context, err := remotecontext.FromArchive(tarStream)

	if err != nil {
		t.Fatalf("Error when creating tar context: %s", err)
	}

	defer func() {
		if err = context.Close(); err != nil {
			t.Fatalf("Error when closing tar context: %s", err)
		}
	}()

	r := strings.NewReader(testCase.dockerfile)
	result, err := parser.Parse(r)

	if err != nil {
		t.Fatalf("Error when parsing Dockerfile: %s", err)
	}

	options := &types.ImageBuildOptions{
		BuildArgs: make(map[string]*string),
	}

	b := &Builder{
		options:   options,
		Stdout:    ioutil.Discard,
		buildArgs: newBuildArgs(options.BuildArgs),
	}

	shlex := NewShellLex(parser.DefaultEscapeToken)
	n := result.AST
	state := &dispatchState{runConfig: &container.Config{}}
	opts := dispatchOptions{
		state:   state,
		stepMsg: formatStep(0, len(n.Children)),
		node:    n.Children[0],
		shlex:   shlex,
		source:  context,
	}
	state, err = b.dispatch(opts)

	if err == nil {
		t.Fatalf("No error when executing test %s", testCase.name)
	}

	if !strings.Contains(err.Error(), testCase.expectedError) {
		t.Fatalf("Wrong error message. Should be \"%s\". Got \"%s\"", testCase.expectedError, err.Error())
	}

}
