/*
Copyright 2014 The Kubernetes Authors.

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

package cp

import (
	"archive/tar"
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/rest/fake"
	kexec "k8s.io/kubectl/pkg/cmd/exec"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

type FileType int

const (
	RegularFile FileType = 0
	SymLink     FileType = 1
	RegexFile   FileType = 2
)

func TestExtractFileSpec(t *testing.T) {
	tests := []struct {
		spec              string
		expectedPod       string
		expectedNamespace string
		expectedFile      string
		expectErr         bool
	}{
		{
			spec:              "namespace/pod:/some/file",
			expectedPod:       "pod",
			expectedNamespace: "namespace",
			expectedFile:      "/some/file",
		},
		{
			spec:         "pod:/some/file",
			expectedPod:  "pod",
			expectedFile: "/some/file",
		},
		{
			spec:         "/some/file",
			expectedFile: "/some/file",
		},
		{
			spec:      ":file:not:exist:in:local:filesystem",
			expectErr: true,
		},
		{
			spec:      "namespace/pod/invalid:/some/file",
			expectErr: true,
		},
		{
			spec:         "pod:/some/filenamewith:in",
			expectedPod:  "pod",
			expectedFile: "/some/filenamewith:in",
		},
	}
	for _, test := range tests {
		spec, err := extractFileSpec(test.spec)
		if test.expectErr && err == nil {
			t.Errorf("unexpected non-error")
			continue
		}
		if err != nil && !test.expectErr {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if spec.PodName != test.expectedPod {
			t.Errorf("expected: %s, saw: %s", test.expectedPod, spec.PodName)
		}
		if spec.PodNamespace != test.expectedNamespace {
			t.Errorf("expected: %s, saw: %s", test.expectedNamespace, spec.PodNamespace)
		}
		specFile := ""
		if spec.File != nil {
			specFile = spec.File.String()
		}
		if specFile != test.expectedFile {
			t.Errorf("expected: %s, saw: %s", test.expectedFile, specFile)
		}
	}
}

func TestGetPrefix(t *testing.T) {
	remoteSeparator := '/'
	osSeparator := os.PathSeparator
	tests := []struct {
		input    string
		expected string
	}{
		{
			input:    "%[1]cfoo%[1]cbar",
			expected: "foo%[1]cbar",
		},
		{
			input:    "foo%[1]cbar",
			expected: "foo%[1]cbar",
		},
	}
	for _, test := range tests {
		outRemote := newRemotePath(fmt.Sprintf(test.input, remoteSeparator)).StripSlashes()
		expectedRemote := fmt.Sprintf(test.expected, remoteSeparator)
		if outRemote.String() != expectedRemote {
			t.Errorf("remote expected: %s, saw: %s", expectedRemote, outRemote.String())
		}
		outLocal := newLocalPath(fmt.Sprintf(test.input, osSeparator)).StripSlashes()
		expectedLocal := fmt.Sprintf(test.expected, osSeparator)
		if outLocal.String() != expectedLocal {
			t.Errorf("local expected: %s, saw: %s", expectedLocal, outLocal.String())
		}
	}
}

func TestStripPathShortcuts(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "test single path shortcut prefix",
			input:    "../foo/bar",
			expected: "foo/bar",
		},
		{
			name:     "test single path shortcut prefix",
			input:    `..\foo\bar`,
			expected: "foo/bar",
		},
		{
			name:     "test multiple path shortcuts",
			input:    "../../foo/bar",
			expected: "foo/bar",
		},
		{
			name:     "test multiple path shortcuts",
			input:    `..\..\foo\bar`,
			expected: "foo/bar",
		},
		{
			name:     "test multiple path shortcuts with absolute path",
			input:    "/tmp/one/two/../../foo/bar",
			expected: "tmp/foo/bar",
		},
		{
			name:     "test multiple path shortcuts with absolute path",
			input:    `\tmp\one\two\..\..\foo\bar`,
			expected: "tmp/foo/bar",
		},
		{
			name:     "test multiple path shortcuts with no named directory",
			input:    "../../",
			expected: "",
		},
		{
			name:     "test multiple path shortcuts with no named directory",
			input:    `..\..\`,
			expected: "",
		},
		{
			name:     "test multiple path shortcuts with no named directory and no trailing slash",
			input:    "../..",
			expected: "",
		},
		{
			name:     "test multiple path shortcuts with no named directory and no trailing slash",
			input:    `..\..`,
			expected: "",
		},
		{
			name:     "test multiple path shortcuts with absolute path and filename containing leading dots",
			input:    "/tmp/one/two/../../foo/..bar",
			expected: "tmp/foo/..bar",
		},
		{
			name:     "test multiple path shortcuts with absolute path and filename containing leading dots",
			input:    `\tmp\one\two\..\..\foo\..bar`,
			expected: "tmp/foo/..bar",
		},
		{
			name:     "test multiple path shortcuts with no named directory and filename containing leading dots",
			input:    "../...foo",
			expected: "...foo",
		},
		{
			name:     "test multiple path shortcuts with no named directory and filename containing leading dots",
			input:    `..\...foo`,
			expected: "...foo",
		},
		{
			name:     "test filename containing leading dots",
			input:    "...foo",
			expected: "...foo",
		},
		{
			name:     "test root directory",
			input:    "/",
			expected: "",
		},
		{
			name:     "test root directory",
			input:    `\`,
			expected: "",
		},
	}

	for i, test := range tests {
		out := newRemotePath(test.input).StripShortcuts()
		if out.String() != test.expected {
			t.Errorf("expected[%d]: %s, saw: %s", i, test.expected, out)
		}
	}
}
func TestIsDestRelative(t *testing.T) {
	tests := []struct {
		base     string
		dest     string
		relative bool
	}{
		{
			base:     "/dir",
			dest:     "/dir/../link",
			relative: false,
		},
		{
			base:     "/dir",
			dest:     "/dir/../../link",
			relative: false,
		},
		{
			base:     "/dir",
			dest:     "/link",
			relative: false,
		},
		{
			base:     "/dir",
			dest:     "/dir/link",
			relative: true,
		},
		{
			base:     "/dir",
			dest:     "/dir/int/../link",
			relative: true,
		},
		{
			base:     "dir",
			dest:     "dir/link",
			relative: true,
		},
		{
			base:     "dir",
			dest:     "dir/int/../link",
			relative: true,
		},
		{
			base:     "dir",
			dest:     "dir/../../link",
			relative: false,
		},
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			if test.relative != isRelative(newLocalPath(test.base), newLocalPath(test.dest)) {
				t.Errorf("unexpected result for: base %q, dest %q", test.base, test.dest)
			}
		})
	}
}

func checkErr(t *testing.T, err error) {
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		t.FailNow()
	}
}

func TestTarUntar(t *testing.T) {
	dir, err := os.MkdirTemp("", "input")
	checkErr(t, err)
	dir = dir + "/"

	dir2, err := os.MkdirTemp("", "output")
	checkErr(t, err)
	dir2 = dir2 + "/"

	dir3, err := os.MkdirTemp("", "dir")
	checkErr(t, err)

	defer func() {
		os.RemoveAll(dir)
		os.RemoveAll(dir2)
		os.RemoveAll(dir3)
	}()

	files := []struct {
		name     string
		nameList []string
		data     string
		omitted  bool
		fileType FileType
	}{
		{
			name:     "foo",
			data:     "foobarbaz",
			fileType: RegularFile,
		},
		{
			name:     "dir/blah",
			data:     "bazblahfoo",
			fileType: RegularFile,
		},
		{
			name:     "some/other/directory",
			data:     "with more data here",
			fileType: RegularFile,
		},
		{
			name:     "blah",
			data:     "same file name different data",
			fileType: RegularFile,
		},
		{
			name:     "gakki",
			data:     "tmp/gakki",
			omitted:  true,
			fileType: SymLink,
		},
		{
			name:     "relative_to_dest",
			data:     dir2 + "/foo",
			omitted:  true,
			fileType: SymLink,
		},
		{
			name:     "tricky_relative",
			data:     dir3 + "/xyz",
			omitted:  true,
			fileType: SymLink,
		},
		{
			name:     "absolute_path",
			data:     "/tmp/gakki",
			omitted:  true,
			fileType: SymLink,
		},
		{
			name:     "blah*",
			nameList: []string{"blah1", "blah2"},
			data:     "regexp file name",
			fileType: RegexFile,
		},
	}

	for _, file := range files {
		completePath := dir + file.name
		if err := os.MkdirAll(filepath.Dir(completePath), 0755); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if file.fileType == RegularFile {
			createTmpFile(t, completePath, file.data)
		} else if file.fileType == SymLink {
			err := os.Symlink(file.data, completePath)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		} else if file.fileType == RegexFile {
			for _, fileName := range file.nameList {
				createTmpFile(t, dir+fileName, file.data)
			}
		} else {
			t.Fatalf("unexpected file type: %v", file)
		}
	}

	opts := NewCopyOptions(genericiooptions.NewTestIOStreamsDiscard())

	writer := &bytes.Buffer{}
	if err := makeTar(newLocalPath(dir), newRemotePath(dir), writer); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	reader := bytes.NewBuffer(writer.Bytes())
	if err := opts.untarAll("", "", "", remotePath{}, newLocalPath(dir2), reader); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, file := range files {
		absPath := dir2 + strings.TrimPrefix(dir, os.TempDir())
		filePath := absPath + file.name

		if file.fileType == RegularFile {
			cmpFileData(t, filePath, file.data)
		} else if file.fileType == SymLink {
			dest, err := os.Readlink(filePath)
			if file.omitted {
				if err != nil && strings.Contains(err.Error(), "no such file or directory") {
					continue
				}
				t.Fatalf("expected to omit symlink for %s", filePath)
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if file.data != dest {
				t.Fatalf("expected: %s, saw: %s", file.data, dest)
			}
		} else if file.fileType == RegexFile {
			for _, fileName := range file.nameList {
				cmpFileData(t, dir+fileName, file.data)
			}
		} else {
			t.Fatalf("unexpected file type: %v", file)
		}
	}
}

func TestTarUntarWrongPrefix(t *testing.T) {
	dir, err := os.MkdirTemp("", "input")
	checkErr(t, err)
	dir = dir + "/"

	dir2, err := os.MkdirTemp("", "output")
	checkErr(t, err)

	defer func() {
		os.RemoveAll(dir)
		os.RemoveAll(dir2)
	}()

	completePath := dir + "foo"
	if err := os.MkdirAll(filepath.Dir(completePath), 0755); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	createTmpFile(t, completePath, "sample data")

	opts := NewCopyOptions(genericiooptions.NewTestIOStreamsDiscard())

	writer := &bytes.Buffer{}
	if err := makeTar(newLocalPath(dir), newRemotePath(dir), writer); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	reader := bytes.NewBuffer(writer.Bytes())
	err = opts.untarAll("", "", "verylongprefix-showing-the-tar-was-tempered-with", remotePath{}, newLocalPath(dir2), reader)
	if err == nil || !strings.Contains(err.Error(), "tar contents corrupted") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestTarDestinationName(t *testing.T) {
	dir, err := os.MkdirTemp(os.TempDir(), "input")
	dir2, err2 := os.MkdirTemp(os.TempDir(), "output")
	if err != nil || err2 != nil {
		t.Errorf("unexpected error: %v | %v", err, err2)
		t.FailNow()
	}
	defer func() {
		if err := os.RemoveAll(dir); err != nil {
			t.Errorf("Unexpected error cleaning up: %v", err)
		}
		if err := os.RemoveAll(dir2); err != nil {
			t.Errorf("Unexpected error cleaning up: %v", err)
		}
	}()

	files := []struct {
		name string
		data string
	}{
		{
			name: "foo",
			data: "foobarbaz",
		},
		{
			name: "dir/blah",
			data: "bazblahfoo",
		},
		{
			name: "some/other/directory",
			data: "with more data here",
		},
		{
			name: "blah",
			data: "same file name different data",
		},
	}

	// ensure files exist on disk
	for _, file := range files {
		completePath := dir + "/" + file.name
		if err := os.MkdirAll(filepath.Dir(completePath), 0755); err != nil {
			t.Errorf("unexpected error: %v", err)
			t.FailNow()
		}
		createTmpFile(t, completePath, file.data)
	}

	reader, writer := io.Pipe()
	go func() {
		if err := makeTar(newLocalPath(dir), newRemotePath(dir2), writer); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}()

	tarReader := tar.NewReader(reader)
	for {
		hdr, err := tarReader.Next()
		if err == io.EOF {
			break
		} else if err != nil {
			t.Errorf("unexpected error: %v", err)
			t.FailNow()
		}

		if !strings.HasPrefix(hdr.Name, filepath.Base(dir2)) {
			t.Errorf("expected %q as destination filename prefix, saw: %q", filepath.Base(dir2), hdr.Name)
		}
	}
}

func TestBadTar(t *testing.T) {
	dir, err := os.MkdirTemp(os.TempDir(), "dest")
	if err != nil {
		t.Errorf("unexpected error: %v ", err)
		t.FailNow()
	}
	defer os.RemoveAll(dir)

	// More or less cribbed from https://golang.org/pkg/archive/tar/#example__minimal
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	var files = []struct {
		name string
		body string
	}{
		{"/prefix/foo/bar/../../home/bburns/names.txt", "Down and back"},
	}
	for _, file := range files {
		hdr := &tar.Header{
			Name: file.name,
			Mode: 0600,
			Size: int64(len(file.body)),
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Errorf("unexpected error: %v ", err)
			t.FailNow()
		}
		if _, err := tw.Write([]byte(file.body)); err != nil {
			t.Errorf("unexpected error: %v ", err)
			t.FailNow()
		}
	}
	if err := tw.Close(); err != nil {
		t.Errorf("unexpected error: %v ", err)
		t.FailNow()
	}

	opts := NewCopyOptions(genericiooptions.NewTestIOStreamsDiscard())
	if err := opts.untarAll("", "", "/prefix", remotePath{}, newLocalPath(dir), &buf); err != nil {
		t.Errorf("unexpected error: %v ", err)
		t.FailNow()
	}

	for _, file := range files {
		_, err := os.Stat(dir + filepath.Clean(file.name[len("/prefix"):]))
		if err != nil {
			t.Errorf("Error finding file: %v", err)
		}
	}
}

func TestCopyToPod(t *testing.T) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	ns := scheme.Codecs.WithoutConversion()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.Client = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Group: "", Version: "v1"},
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			responsePod := &v1.Pod{}
			return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(codec, responsePod))))}, nil
		}),
	}

	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
	ioStreams, _, _, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdCp(tf, ioStreams)

	srcFile, err := os.MkdirTemp("", "test")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		t.FailNow()
	}
	defer os.RemoveAll(srcFile)

	tests := map[string]struct {
		src         string
		dest        string
		expectedErr bool
	}{
		"copy to directory": {
			src:         srcFile,
			dest:        "/tmp/",
			expectedErr: false,
		},
		"copy to root": {
			src:         srcFile,
			dest:        "/",
			expectedErr: false,
		},
		"copy to empty file name": {
			src:         srcFile,
			dest:        "",
			expectedErr: true,
		},
		"copy unexisting file": {
			src:         filepath.Join(srcFile, "nope"),
			dest:        "/tmp",
			expectedErr: true,
		},
	}

	for name, test := range tests {
		opts := NewCopyOptions(ioStreams)
		opts.Complete(tf, cmd, []string{test.src, fmt.Sprintf("pod-ns/pod-name:%s", test.dest)})
		t.Run(name, func(t *testing.T) {
			err = opts.Run()
			//If error is NotFound error , it indicates that the
			//request has been sent correctly.
			//Treat this as no error.
			if test.expectedErr && errors.IsNotFound(err) {
				t.Errorf("expected error but didn't get one")
			}
			if !test.expectedErr && !errors.IsNotFound(err) {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestCopyToPodNoPreserve(t *testing.T) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	ns := scheme.Codecs.WithoutConversion()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.Client = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Group: "", Version: "v1"},
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			responsePod := &v1.Pod{}
			return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(codec, responsePod))))}, nil
		}),
	}

	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
	ioStreams, _, _, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdCp(tf, ioStreams)

	srcFile, err := os.MkdirTemp("", "test")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		t.FailNow()
	}
	defer os.RemoveAll(srcFile)

	tests := map[string]struct {
		expectedCmd []string
		nopreserve  bool
	}{
		"copy to pod no preserve user and permissions": {
			expectedCmd: []string{"tar", "--no-same-permissions", "--no-same-owner", "-xmf", "-", "-C", "."},
			nopreserve:  true,
		},
		"copy to pod preserve user and permissions": {
			expectedCmd: []string{"tar", "-xmf", "-", "-C", "."},
			nopreserve:  false,
		},
	}
	opts := NewCopyOptions(ioStreams)
	src := fileSpec{
		File: newLocalPath(srcFile),
	}
	dest := fileSpec{
		PodNamespace: "pod-ns",
		PodName:      "pod-name",
		File:         newRemotePath("foo"),
	}
	opts.Complete(tf, cmd, nil)

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			options := &kexec.ExecOptions{}
			opts.NoPreserve = test.nopreserve
			err = opts.copyToPod(src, dest, options)
			if !(reflect.DeepEqual(test.expectedCmd, options.Command)) {
				t.Errorf("expected cmd: %v, got: %v", test.expectedCmd, options.Command)
			}
		})
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		expectedErr bool
	}{
		{
			name:        "Validate Succeed",
			args:        []string{"1", "2"},
			expectedErr: false,
		},
		{
			name:        "Validate Fail",
			args:        []string{"1", "2", "3"},
			expectedErr: true,
		},
	}
	ioStreams, _, _, _ := genericiooptions.NewTestIOStreams()
	opts := NewCopyOptions(ioStreams)

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			opts.args = test.args
			err := opts.Validate()
			if (err != nil) != test.expectedErr {
				t.Errorf("expected error: %v, saw: %v, error: %v", test.expectedErr, err != nil, err)
			}
		})
	}
}

func TestUntar(t *testing.T) {
	testdir, err := os.MkdirTemp("", "test-untar")
	require.NoError(t, err)
	defer os.RemoveAll(testdir)
	t.Logf("Test base: %s", testdir)

	basedir := testdir + "/" + "base"

	type file struct {
		path       string
		linkTarget string // For link types
		expected   string // Expect to find the file here (or not, if empty)
	}
	files := []file{{
		// Absolute file within dest
		path:     basedir + "/" + "abs",
		expected: basedir + basedir + "/" + "abs",
	}, { // Absolute file outside dest
		path:     testdir + "/" + "abs-out",
		expected: basedir + testdir + "/" + "abs-out",
	}, { // Absolute nested file within dest
		path:     basedir + "/" + "nested/nest-abs",
		expected: basedir + basedir + "/" + "nested/nest-abs",
	}, { // Absolute nested file outside dest
		path:     basedir + "/" + "nested/../../nest-abs-out",
		expected: basedir + testdir + "/" + "nest-abs-out",
	}, { // Relative file inside dest
		path:     "relative",
		expected: basedir + "/" + "relative",
	}, { // Relative file outside dest
		path:     "../unrelative",
		expected: "",
	}, { // Relative file outside dest (windows)
		path:     `..\unrelative-windows`,
		expected: "",
	}, { // Nested relative file inside dest
		path:     "nested/nest-rel",
		expected: basedir + "/" + "nested/nest-rel",
	}, { // Nested relative file outside dest
		path:     "nested/../../nest-unrelative",
		expected: "",
	}, { // Nested relative file outside dest (windows)
		path:     `nested\..\..\nest-unrelative`,
		expected: "",
	}}

	links := []file{}
	for _, f := range files {
		links = append(links, file{
			path:       f.path + "-innerlink",
			linkTarget: "link-target",
			expected:   "",
		}, file{
			path:       f.path + "-innerlink-abs",
			linkTarget: basedir + "/" + "link-target",
			expected:   "",
		}, file{
			path:       f.path + "-backlink",
			linkTarget: ".." + "/" + "link-target",
			expected:   "",
		}, file{
			path:       f.path + "-outerlink-abs",
			linkTarget: testdir + "/" + "link-target",
			expected:   "",
		})

		if f.expected != "" {
			// outerlink is the number of backticks to escape to testdir
			outerlink, _ := filepath.Rel(f.expected, testdir)
			links = append(links, file{
				path:       f.path + "outerlink",
				linkTarget: outerlink + "/" + "link-target",
				expected:   "",
			})
		}
	}
	files = append(files, links...)

	// Test back-tick escaping through a symlink.
	files = append(files,
		file{
			path:       "nested/again/back-link",
			linkTarget: "../../nested",
			expected:   "",
		},
		file{
			path:     "nested/again/back-link/../../../back-link-file",
			expected: basedir + "/" + "back-link-file",
		})

	// Test chaining back-tick symlinks.
	files = append(files,
		file{
			path:       "nested/back-link-first",
			linkTarget: "../",
			expected:   "",
		},
		file{
			path:       "nested/back-link-first/back-link-second",
			linkTarget: "../",
			expected:   "",
		})

	files = append(files,
		file{ // Relative directory path with terminating /
			path:     "direct/dir/",
			expected: "",
		})

	buf := &bytes.Buffer{}
	tw := tar.NewWriter(buf)
	expectations := map[string]bool{}
	for _, f := range files {
		if f.expected != "" {
			expectations[f.expected] = false
		}
		if f.linkTarget == "" {
			hdr := &tar.Header{
				Name: f.path,
				Mode: 0666,
				Size: int64(len(f.path)),
			}
			require.NoError(t, tw.WriteHeader(hdr), f.path)
			if !strings.HasSuffix(f.path, "/") {
				_, err := tw.Write([]byte(f.path))
				require.NoError(t, err, f.path)
			}
		} else {
			hdr := &tar.Header{
				Name:     f.path,
				Mode:     int64(0777 | os.ModeSymlink),
				Typeflag: tar.TypeSymlink,
				Linkname: f.linkTarget,
			}
			require.NoError(t, tw.WriteHeader(hdr), f.path)
		}
	}
	tw.Close()

	// Capture warnings to stderr for debugging.
	output := (*testWriter)(t)
	opts := NewCopyOptions(genericiooptions.IOStreams{In: &bytes.Buffer{}, Out: output, ErrOut: output})

	require.NoError(t, opts.untarAll("", "", "", remotePath{}, newLocalPath(basedir), buf))

	filepath.Walk(testdir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil // Ignore directories.
		}
		if _, ok := expectations[path]; !ok {
			t.Errorf("Unexpected file at %s", path)
		} else {
			expectations[path] = true
		}
		return nil
	})
	for path, found := range expectations {
		if !found {
			t.Errorf("Missing expected file %s", path)
		}
	}
}

func TestUntar_SingleFile(t *testing.T) {
	testdir, err := os.MkdirTemp("", "test-untar")
	require.NoError(t, err)
	defer os.RemoveAll(testdir)

	dest := testdir + "/" + "target"

	buf := &bytes.Buffer{}
	tw := tar.NewWriter(buf)

	const (
		srcName = "source"
		content = "file contents"
	)
	hdr := &tar.Header{
		Name: srcName,
		Mode: 0666,
		Size: int64(len(content)),
	}
	require.NoError(t, tw.WriteHeader(hdr))
	_, err = tw.Write([]byte(content))
	require.NoError(t, err)
	tw.Close()

	// Capture warnings to stderr for debugging.
	output := (*testWriter)(t)
	opts := NewCopyOptions(genericiooptions.IOStreams{In: &bytes.Buffer{}, Out: output, ErrOut: output})

	require.NoError(t, opts.untarAll("", "", srcName, remotePath{}, newLocalPath(dest), buf))
	cmpFileData(t, dest, content)
}

func createTmpFile(t *testing.T, filepath, data string) {
	f, err := os.Create(filepath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer f.Close()
	if _, err := io.Copy(f, bytes.NewBuffer([]byte(data))); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func cmpFileData(t *testing.T, filePath, data string) {
	actual, err := os.ReadFile(filePath)
	require.NoError(t, err)
	assert.EqualValues(t, data, actual)
}

type testWriter testing.T

func (t *testWriter) Write(p []byte) (n int, err error) {
	t.Logf(string(p))
	return len(p), nil
}
