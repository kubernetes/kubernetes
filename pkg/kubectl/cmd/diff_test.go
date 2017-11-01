/*
Copyright 2017 The Kubernetes Authors.

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

package cmd

import (
	"bytes"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strings"
	"testing"

	"k8s.io/utils/exec"
)

type FakeObject struct {
	name   string
	local  map[string]interface{}
	merged map[string]interface{}
	live   map[string]interface{}
	last   map[string]interface{}
}

var _ Object = &FakeObject{}

func (f *FakeObject) Name() string {
	return f.name
}

func (f *FakeObject) Local() (map[string]interface{}, error) {
	return f.local, nil
}

func (f *FakeObject) Merged() (map[string]interface{}, error) {
	return f.merged, nil
}

func (f *FakeObject) Live() (map[string]interface{}, error) {
	return f.live, nil
}

func (f *FakeObject) Last() (map[string]interface{}, error) {
	return f.last, nil
}

func TestArguments(t *testing.T) {
	tests := []struct {
		// Input
		args []string

		// Outputs
		from string
		to   string
		err  string
	}{
		// Defaults
		{
			args: []string{},
			from: "LOCAL",
			to:   "LIVE",
			err:  "",
		},
		// One valid argument
		{
			args: []string{"MERGED"},
			from: "MERGED",
			to:   "LIVE",
			err:  "",
		},
		// One invalid argument
		{
			args: []string{"WRONG"},
			from: "",
			to:   "",
			err:  `Invalid parameter "WRONG", must be either "LOCAL", "LIVE", "LAST" or "MERGED"`,
		},
		// Two valid arguments
		{
			args: []string{"MERGED", "LAST"},
			from: "MERGED",
			to:   "LAST",
			err:  "",
		},
		// Two same arguments is fine
		{
			args: []string{"MERGED", "MERGED"},
			from: "MERGED",
			to:   "MERGED",
			err:  "",
		},
		// Second argument is invalid
		{
			args: []string{"MERGED", "WRONG"},
			from: "",
			to:   "",
			err:  `Invalid parameter "WRONG", must be either "LOCAL", "LIVE", "LAST" or "MERGED"`,
		},
		// Three arguments
		{
			args: []string{"MERGED", "LIVE", "LAST"},
			from: "",
			to:   "",
			err:  `Invalid number of arguments: expected at most 2.`,
		},
	}

	for _, test := range tests {
		from, to, e := parseDiffArguments(test.args)
		err := ""
		if e != nil {
			err = e.Error()
		}
		if from != test.from || to != test.to || err != test.err {
			t.Errorf("parseDiffArguments(%v) = (%v, %v, %v), expected (%v, %v, %v)",
				test.args,
				from, to, err,
				test.from, test.to, test.err,
			)
		}
	}
}

func TestDiffProgram(t *testing.T) {
	os.Setenv("KUBERNETES_EXTERNAL_DIFF", "echo")
	stdout := bytes.Buffer{}
	diff := DiffProgram{
		Stdout: &stdout,
		Stderr: &bytes.Buffer{},
		Exec:   exec.New(),
	}
	err := diff.Run("one", "two")
	if err != nil {
		t.Fatal(err)
	}
	if output := stdout.String(); output != "one two\n" {
		t.Fatalf(`stdout = %q, expected "one two\n"`, output)
	}
}

func TestPrinter(t *testing.T) {
	printer := Printer{}

	obj := map[string]interface{}{
		"string": "string",
		"list":   []int{1, 2, 3},
		"int":    12,
	}
	buf := bytes.Buffer{}
	printer.Print(obj, &buf)
	want := `int: 12
list:
- 1
- 2
- 3
string: string
`
	if buf.String() != want {
		t.Errorf("Print() = %q, want %q", buf.String(), want)
	}
}

func TestDiffVersion(t *testing.T) {
	diff, err := NewDiffVersion("LOCAL")
	if err != nil {
		t.Fatal(err)
	}
	defer diff.Dir.Delete()

	obj := FakeObject{
		name:   "bla",
		local:  map[string]interface{}{"local": true},
		last:   map[string]interface{}{"last": true},
		live:   map[string]interface{}{"live": true},
		merged: map[string]interface{}{"merged": true},
	}
	err = diff.Print(&obj, Printer{})
	if err != nil {
		t.Fatal(err)
	}
	fcontent, err := ioutil.ReadFile(path.Join(diff.Dir.Name, obj.Name()))
	if err != nil {
		t.Fatal(err)
	}
	econtent := "local: true\n"
	if string(fcontent) != econtent {
		t.Fatalf("File has %q, expected %q", string(fcontent), econtent)
	}
}

func TestDirectory(t *testing.T) {
	dir, err := CreateDirectory("prefix")
	defer dir.Delete()
	if err != nil {
		t.Fatal(err)
	}
	_, err = os.Stat(dir.Name)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(filepath.Base(dir.Name), "prefix") {
		t.Fatalf(`Directory doesn't start with "prefix": %q`, dir.Name)
	}
	entries, err := ioutil.ReadDir(dir.Name)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("Directory should be empty, has %d elements", len(entries))
	}
	_, err = dir.NewFile("ONE")
	if err != nil {
		t.Fatal(err)
	}
	_, err = dir.NewFile("TWO")
	if err != nil {
		t.Fatal(err)
	}
	entries, err = ioutil.ReadDir(dir.Name)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("ReadDir should have two elements, has %d elements", len(entries))
	}
	err = dir.Delete()
	if err != nil {
		t.Fatal(err)
	}
	_, err = os.Stat(dir.Name)
	if err == nil {
		t.Fatal("Directory should be gone, still present.")
	}
}

func TestDiffer(t *testing.T) {
	diff, err := NewDiffer("LOCAL", "LIVE")
	if err != nil {
		t.Fatal(err)
	}
	defer diff.TearDown()

	obj := FakeObject{
		name:   "bla",
		local:  map[string]interface{}{"local": true},
		last:   map[string]interface{}{"last": true},
		live:   map[string]interface{}{"live": true},
		merged: map[string]interface{}{"merged": true},
	}
	err = diff.Diff(&obj, Printer{})
	if err != nil {
		t.Fatal(err)
	}
	fcontent, err := ioutil.ReadFile(path.Join(diff.From.Dir.Name, obj.Name()))
	if err != nil {
		t.Fatal(err)
	}
	econtent := "local: true\n"
	if string(fcontent) != econtent {
		t.Fatalf("File has %q, expected %q", string(fcontent), econtent)
	}

	fcontent, err = ioutil.ReadFile(path.Join(diff.To.Dir.Name, obj.Name()))
	if err != nil {
		t.Fatal(err)
	}
	econtent = "live: true\n"
	if string(fcontent) != econtent {
		t.Fatalf("File has %q, expected %q", string(fcontent), econtent)
	}
}
