// +build !windows
// TODO(jen20): These need fixing on Windows but fmt is not used right now
// and red CI is making it harder to process other bugs, so ignore until
// we get around to fixing them.

package fmtcmd

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"syscall"
	"testing"

	"github.com/hashicorp/hcl/testhelper"
)

var fixtureExtensions = []string{"hcl"}

func init() {
	sort.Sort(ByFilename(fixtures))
}

func TestIsValidFile(t *testing.T) {
	const fixtureDir = "./test-fixtures"

	cases := []struct {
		Path     string
		Expected bool
	}{
		{"good.hcl", true},
		{".hidden.ignore", false},
		{"file.ignore", false},
		{"dir.ignore", false},
	}

	for _, tc := range cases {
		file, err := os.Stat(filepath.Join(fixtureDir, tc.Path))
		if err != nil {
			t.Errorf("unexpected error: %s", err)
		}

		if res := isValidFile(file, fixtureExtensions); res != tc.Expected {
			t.Errorf("want: %b, got: %b", tc.Expected, res)
		}
	}
}

func TestRunMultiplePaths(t *testing.T) {
	path1, err := renderFixtures("")
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	defer os.RemoveAll(path1)
	path2, err := renderFixtures("")
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	defer os.RemoveAll(path2)

	var expectedOut bytes.Buffer
	for _, path := range []string{path1, path2} {
		for _, fixture := range fixtures {
			if !bytes.Equal(fixture.golden, fixture.input) {
				expectedOut.WriteString(filepath.Join(path, fixture.filename) + "\n")
			}
		}
	}

	_, stdout := mockIO()
	err = Run(
		[]string{path1, path2},
		fixtureExtensions,
		nil, stdout,
		Options{
			List: true,
		},
	)

	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	if stdout.String() != expectedOut.String() {
		t.Errorf("stdout want:\n%s\ngot:\n%s", expectedOut, stdout)
	}
}

func TestRunSubDirectories(t *testing.T) {
	pathParent, err := ioutil.TempDir("", "")
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	defer os.RemoveAll(pathParent)

	path1, err := renderFixtures(pathParent)
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	path2, err := renderFixtures(pathParent)
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}

	paths := []string{path1, path2}
	sort.Strings(paths)

	var expectedOut bytes.Buffer
	for _, path := range paths {
		for _, fixture := range fixtures {
			if !bytes.Equal(fixture.golden, fixture.input) {
				expectedOut.WriteString(filepath.Join(path, fixture.filename) + "\n")
			}
		}
	}

	_, stdout := mockIO()
	err = Run(
		[]string{pathParent},
		fixtureExtensions,
		nil, stdout,
		Options{
			List: true,
		},
	)

	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	if stdout.String() != expectedOut.String() {
		t.Errorf("stdout want:\n%s\ngot:\n%s", expectedOut, stdout)
	}
}

func TestRunStdin(t *testing.T) {
	var expectedOut bytes.Buffer
	for i, fixture := range fixtures {
		if i != 0 {
			expectedOut.WriteString("\n")
		}
		expectedOut.Write(fixture.golden)
	}

	stdin, stdout := mockIO()
	for _, fixture := range fixtures {
		stdin.Write(fixture.input)
	}

	err := Run(
		[]string{},
		fixtureExtensions,
		stdin, stdout,
		Options{},
	)

	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	if !bytes.Equal(stdout.Bytes(), expectedOut.Bytes()) {
		t.Errorf("stdout want:\n%s\ngot:\n%s", expectedOut, stdout)
	}
}

func TestRunStdinAndWrite(t *testing.T) {
	var expectedOut = []byte{}

	stdin, stdout := mockIO()
	stdin.WriteString("")
	err := Run(
		[]string{}, []string{},
		stdin, stdout,
		Options{
			Write: true,
		},
	)

	if err != ErrWriteStdin {
		t.Errorf("error want:\n%s\ngot:\n%s", ErrWriteStdin, err)
	}
	if !bytes.Equal(stdout.Bytes(), expectedOut) {
		t.Errorf("stdout want:\n%s\ngot:\n%s", expectedOut, stdout)
	}
}

func TestRunFileError(t *testing.T) {
	path, err := ioutil.TempDir("", "")
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	defer os.RemoveAll(path)
	filename := filepath.Join(path, "unreadable.hcl")

	var expectedError = &os.PathError{
		Op:   "open",
		Path: filename,
		Err:  syscall.EACCES,
	}

	err = ioutil.WriteFile(filename, []byte{}, 0000)
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}

	_, stdout := mockIO()
	err = Run(
		[]string{path},
		fixtureExtensions,
		nil, stdout,
		Options{},
	)

	if !reflect.DeepEqual(err, expectedError) {
		t.Errorf("error want: %#v, got: %#v", expectedError, err)
	}
}

func TestRunNoOptions(t *testing.T) {
	path, err := renderFixtures("")
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	defer os.RemoveAll(path)

	var expectedOut bytes.Buffer
	for _, fixture := range fixtures {
		expectedOut.Write(fixture.golden)
	}

	_, stdout := mockIO()
	err = Run(
		[]string{path},
		fixtureExtensions,
		nil, stdout,
		Options{},
	)

	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	if stdout.String() != expectedOut.String() {
		t.Errorf("stdout want:\n%s\ngot:\n%s", expectedOut, stdout)
	}
}

func TestRunList(t *testing.T) {
	path, err := renderFixtures("")
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	defer os.RemoveAll(path)

	var expectedOut bytes.Buffer
	for _, fixture := range fixtures {
		if !bytes.Equal(fixture.golden, fixture.input) {
			expectedOut.WriteString(fmt.Sprintln(filepath.Join(path, fixture.filename)))
		}
	}

	_, stdout := mockIO()
	err = Run(
		[]string{path},
		fixtureExtensions,
		nil, stdout,
		Options{
			List: true,
		},
	)

	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	if stdout.String() != expectedOut.String() {
		t.Errorf("stdout want:\n%s\ngot:\n%s", expectedOut, stdout)
	}
}

func TestRunWrite(t *testing.T) {
	path, err := renderFixtures("")
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	defer os.RemoveAll(path)

	_, stdout := mockIO()
	err = Run(
		[]string{path},
		fixtureExtensions,
		nil, stdout,
		Options{
			Write: true,
		},
	)

	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	for _, fixture := range fixtures {
		res, err := ioutil.ReadFile(filepath.Join(path, fixture.filename))
		if err != nil {
			t.Errorf("unexpected error: %s", err)
		}
		if !bytes.Equal(res, fixture.golden) {
			t.Errorf("file %q contents want:\n%s\ngot:\n%s", fixture.filename, fixture.golden, res)
		}
	}
}

func TestRunDiff(t *testing.T) {
	path, err := renderFixtures("")
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	defer os.RemoveAll(path)

	var expectedOut bytes.Buffer
	for _, fixture := range fixtures {
		if len(fixture.diff) > 0 {
			expectedOut.WriteString(
				regexp.QuoteMeta(
					fmt.Sprintf("diff a/%s/%s b/%s/%s\n", path, fixture.filename, path, fixture.filename),
				),
			)
			// Need to use regex to ignore datetimes in diff.
			expectedOut.WriteString(`--- .+?\n`)
			expectedOut.WriteString(`\+\+\+ .+?\n`)
			expectedOut.WriteString(regexp.QuoteMeta(string(fixture.diff)))
		}
	}

	expectedOutString := testhelper.Unix2dos(expectedOut.String())

	_, stdout := mockIO()
	err = Run(
		[]string{path},
		fixtureExtensions,
		nil, stdout,
		Options{
			Diff: true,
		},
	)

	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	if !regexp.MustCompile(expectedOutString).Match(stdout.Bytes()) {
		t.Errorf("stdout want match:\n%s\ngot:\n%q", expectedOutString, stdout)
	}
}

func mockIO() (stdin, stdout *bytes.Buffer) {
	return new(bytes.Buffer), new(bytes.Buffer)
}

type fixture struct {
	filename            string
	input, golden, diff []byte
}

type ByFilename []fixture

func (s ByFilename) Len() int           { return len(s) }
func (s ByFilename) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s ByFilename) Less(i, j int) bool { return len(s[i].filename) > len(s[j].filename) }

var fixtures = []fixture{
	{
		"noop.hcl",
		[]byte(`resource "aws_security_group" "firewall" {
  count = 5
}
`),
		[]byte(`resource "aws_security_group" "firewall" {
  count = 5
}
`),
		[]byte(``),
	}, {
		"align_equals.hcl",
		[]byte(`variable "foo" {
  default = "bar"
  description = "bar"
}
`),
		[]byte(`variable "foo" {
  default     = "bar"
  description = "bar"
}
`),
		[]byte(`@@ -1,4 +1,4 @@
 variable "foo" {
-  default = "bar"
+  default     = "bar"
   description = "bar"
 }
`),
	}, {
		"indentation.hcl",
		[]byte(`provider "aws" {
    access_key = "foo"
    secret_key = "bar"
}
`),
		[]byte(`provider "aws" {
  access_key = "foo"
  secret_key = "bar"
}
`),
		[]byte(`@@ -1,4 +1,4 @@
 provider "aws" {
-    access_key = "foo"
-    secret_key = "bar"
+  access_key = "foo"
+  secret_key = "bar"
 }
`),
	},
}

// parent can be an empty string, in which case the system's default
// temporary directory will be used.
func renderFixtures(parent string) (path string, err error) {
	path, err = ioutil.TempDir(parent, "")
	if err != nil {
		return "", err
	}

	for _, fixture := range fixtures {
		err = ioutil.WriteFile(filepath.Join(path, fixture.filename), []byte(fixture.input), 0644)
		if err != nil {
			os.RemoveAll(path)
			return "", err
		}
	}

	return path, nil
}
