/*
Copyright 2014 Google Inc. All rights reserved.

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

package volume

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
)

type MockDiskUtil struct{}

// TODO(jonesdl) To fully test this, we could create a loopback device
// and mount that instead.
func (util *MockDiskUtil) AttachDisk(PD *GCEPersistentDisk) error {
	err := os.MkdirAll(path.Join(PD.RootDir, "global", "pd", PD.PDName), 0750)
	if err != nil {
		return err
	}
	return nil
}

func (util *MockDiskUtil) DetachDisk(PD *GCEPersistentDisk, devicePath string) error {
	err := os.RemoveAll(path.Join(PD.RootDir, "global", "pd", PD.PDName))
	if err != nil {
		return err
	}
	return nil
}

type MockMounter struct{}

func (mounter *MockMounter) Mount(source string, target string, fstype string, flags uintptr, data string) error {
	return nil
}

func (mounter *MockMounter) Unmount(target string, flags int) error {
	return nil
}

func (mounter *MockMounter) RefCount(vol Interface) (string, int, error) {
	return "", 0, nil
}

func TestCreateVolumeBuilders(t *testing.T) {
	tempDir := "CreateVolumes"
	createVolumesTests := []struct {
		volume api.Volume
		path   string
		podID  string
	}{
		{
			api.Volume{
				Name: "host-dir",
				Source: &api.VolumeSource{
					HostDir: &api.HostDir{"/dir/path"},
				},
			},
			"/dir/path",
			"",
		},
		{
			api.Volume{
				Name: "empty-dir",
				Source: &api.VolumeSource{
					EmptyDir: &api.EmptyDir{},
				},
			},
			path.Join(tempDir, "/my-id/volumes/empty/empty-dir"),
			"my-id",
		},
		{
			api.Volume{
				Name: "gce-pd",
				Source: &api.VolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDisk{"my-disk", "ext4", 0, false},
				},
			},
			path.Join(tempDir, "/my-id/volumes/gce-pd/gce-pd"),
			"my-id",
		},
		{api.Volume{}, "", ""},
		{
			api.Volume{
				Name:   "empty-dir",
				Source: &api.VolumeSource{},
			},
			"",
			"",
		},
	}
	for _, createVolumesTest := range createVolumesTests {
		tt := createVolumesTest
		vb, err := CreateVolumeBuilder(&tt.volume, tt.podID, tempDir)
		if tt.volume.Source == nil {
			if vb != nil {
				t.Errorf("Expected volume to be nil")
			}
			continue
		}
		if tt.volume.Source.HostDir == nil && tt.volume.Source.EmptyDir == nil && tt.volume.Source.GCEPersistentDisk == nil {
			if err != ErrUnsupportedVolumeType {
				t.Errorf("Unexpected error: %v", err)
			}
			continue
		}
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		path := vb.GetPath()
		if path != tt.path {
			t.Errorf("Unexpected bind path. Expected %v, got %v", tt.path, path)
		}
	}
}

func TestCreateVolumeCleaners(t *testing.T) {
	tempDir := "CreateVolumeCleaners"
	createVolumeCleanerTests := []struct {
		kind  string
		name  string
		podID string
	}{
		{"empty", "empty-vol", "my-id"},
		{"", "", ""},
		{"gce-pd", "gce-pd-vol", "my-id"},
	}
	for _, tt := range createVolumeCleanerTests {
		vol, err := CreateVolumeCleaner(tt.kind, tt.name, tt.podID, tempDir)
		if tt.kind == "" && err != nil && vol == nil {
			continue
		}
		if err != nil {
			t.Errorf("Unexpected error occured: %v", err)
		}
		actualKind := reflect.TypeOf(vol).Elem().Name()
		if tt.kind == "empty" && actualKind != "EmptyDir" {
			t.Errorf("CreateVolumeCleaner returned invalid type. Expected EmptyDirectory, got %v, %v", tt.kind, actualKind)
		}
		if tt.kind == "gce-pd" && actualKind != "GCEPersistentDisk" {
			t.Errorf("CreateVolumeCleaner returned invalid type. Expected PersistentDisk, got %v, %v", tt.kind, actualKind)
		}
	}
}

func TestSetUpAndTearDown(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "CreateVolumes")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	defer os.RemoveAll(tempDir)
	fakeID := "my-id"
	type VolumeTester interface {
		Builder
		Cleaner
	}
	volumes := []VolumeTester{
		&EmptyDir{"empty", fakeID, tempDir},
		&GCEPersistentDisk{"pd", fakeID, tempDir, "pd-disk", "ext4", "", false, &MockDiskUtil{}, &MockMounter{}},
	}

	for _, vol := range volumes {
		err = vol.SetUp()
		path := vol.GetPath()
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %v", path)
		}
		err = vol.TearDown()
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Errorf("TearDown() failed, original volume path not properly removed: %v", path)
		}
	}
}

func TestGetActiveVolumes(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "CreateVolumes")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	defer os.RemoveAll(tempDir)
	getActiveVolumesTests := []struct {
		name       string
		podID      string
		kind       string
		identifier string
	}{
		{"fakeName", "fakeID", "empty", "fakeID/fakeName"},
		{"fakeName2", "fakeID2", "empty", "fakeID2/fakeName2"},
	}
	expectedIdentifiers := []string{}
	for _, test := range getActiveVolumesTests {
		volumeDir := path.Join(tempDir, test.podID, "volumes", test.kind, test.name)
		os.MkdirAll(volumeDir, 0750)
		expectedIdentifiers = append(expectedIdentifiers, test.identifier)
	}
	volumeMap := GetCurrentVolumes(tempDir)
	for _, name := range expectedIdentifiers {
		if _, ok := volumeMap[name]; !ok {
			t.Errorf("Expected volume map entry not found: %v", name)
		}
	}
}

type fakeExec struct {
	cmds   [][]string
	dirs   []string
	data   []byte
	err    error
	action func([]string, string)
}

func (f *fakeExec) ExecCommand(cmd []string, dir string) ([]byte, error) {
	f.cmds = append(f.cmds, cmd)
	f.dirs = append(f.dirs, dir)
	f.action(cmd, dir)
	return f.data, f.err
}

func TestGitVolume(t *testing.T) {
	var fcmd exec.FakeCmd
	fcmd = exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			func() ([]byte, error) {
				os.MkdirAll(path.Join(fcmd.Dirs[0], "kubernetes"), 0750)
				return []byte{}, nil
			},
			func() ([]byte, error) { return []byte{}, nil },
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fake := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	dir := os.TempDir() + "/git"
	g := GitDir{
		Source:   "https://github.com/GoogleCloudPlatform/kubernetes.git",
		Revision: "2a30ce65c5ab586b98916d83385c5983edd353a1",
		PodID:    "foo",
		RootDir:  dir,
		Name:     "test-pod",
		exec:     &fake,
	}
	err := g.SetUp()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	expectedCmds := [][]string{
		{"git", "clone", g.Source},
		{"git", "checkout", g.Revision},
		{"git", "reset", "--hard"},
	}
	if fake.CommandCalls != len(expectedCmds) {
		t.Errorf("unexpected command calls: expected 3, saw: %d", fake.CommandCalls)
	}
	if !reflect.DeepEqual(expectedCmds, fcmd.CombinedOutputLog) {
		t.Errorf("unexpected commands: %v, expected: %v", fcmd.CombinedOutputLog, expectedCmds)
	}
	expectedDirs := []string{g.GetPath(), g.GetPath() + "/kubernetes", g.GetPath() + "/kubernetes"}
	if len(fcmd.Dirs) != 3 || !reflect.DeepEqual(expectedDirs, fcmd.Dirs) {
		t.Errorf("unexpected directories: %v, expected: %v", fcmd.Dirs, expectedDirs)
	}
	err = g.TearDown()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

type DataHandler struct {
	data        []byte
	contentType string
	statusCode  int
	t           *testing.T
}

func (d *DataHandler) ServeHTTP(resp http.ResponseWriter, req *http.Request) {
	resp.WriteHeader(d.statusCode)
	resp.Header().Set("Content-type", d.contentType)
	_, err := resp.Write(d.data)
	if err != nil {
		d.t.Errorf("unexpected error: %v", err)
	}
}

type fileNameAndContents struct {
	fileName string
	contents string
}

func makeZip(contents []fileNameAndContents) ([]byte, error) {
	buf := new(bytes.Buffer)
	writer := zip.NewWriter(buf)
	for _, f := range contents {
		file, err := writer.Create(f.fileName)
		if err != nil {
			return nil, err
		}
		if _, err = file.Write([]byte(f.contents)); err != nil {
			return nil, err
		}
	}
	err := writer.Close()
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func makeTarball(contents []fileNameAndContents) ([]byte, error) {
	buf := new(bytes.Buffer)
	writer := tar.NewWriter(buf)
	for _, f := range contents {
		header := &tar.Header{
			Name: f.fileName,
			Size: int64(len(f.contents)),
		}
		if err := writer.WriteHeader(header); err != nil {
			return nil, err
		}
		if _, err := writer.Write([]byte(f.contents)); err != nil {
			return nil, err
		}
	}
	err := writer.Close()
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func TestHTTPVolume(t *testing.T) {
	tests := []struct {
		files       []fileNameAndContents
		contentType string
		path        string
		code        int
	}{
		{[]fileNameAndContents{{"test.txt", "this is a test file"}}, "text/plain", "/", http.StatusOK},
		{[]fileNameAndContents{{"test2.txt", "this is a different test"}}, "text/plain", "/some/path/here/", http.StatusOK},
		{[]fileNameAndContents{{"error.txt", "nothing here"}}, "text/plain", "/some/error/", http.StatusInternalServerError},
		{[]fileNameAndContents{{"testA.txt", "this is an A archive test"}, {"testB.txt", "this is an B archive test"}}, "application/x-compressed", "/some/path/here/", http.StatusOK},
		{[]fileNameAndContents{{"testZipA.txt", "this is an zip A archive test"}, {"testZipB.txt", "this is an zip B archive test"}}, "application/zip", "/some/zippy/path/here/", http.StatusOK},
	}
	for _, test := range tests {
		var fileName string
		var handler http.Handler
		var archive bool
		if len(test.files) == 1 {
			handler = &DataHandler{
				data:        []byte(test.files[0].contents),
				contentType: test.contentType,
				statusCode:  test.code,
				t:           t,
			}
			fileName = test.files[0].fileName
			archive = false
		} else {
			var err error
			var data []byte
			if isZip("", test.contentType) {
				if data, err = makeZip(test.files); err != nil {
					t.Fatalf("failed to build zip: %v", err)
				}
				fileName = "archive.zip"
			} else if isTarball("", test.contentType) {
				if data, err = makeTarball(test.files); err != nil {
					t.Fatalf("failed to build tar: %v", err)
				}
				fileName = "archive.tgz"
			} else {
				t.Fatalf("unexpected archive content type: %s", test.contentType)
			}
			handler = &DataHandler{
				data:        data,
				contentType: test.contentType,
				statusCode:  test.code,
				t:           t,
			}
			archive = true
		}
		server := httptest.NewServer(handler)
		dir := os.TempDir() + "/http"
		h := HTTPDir{
			URL:     server.URL + test.path + fileName,
			RootDir: dir,
			Name:    "test",
			PodID:   "foo",
			Archive: archive,
		}
		err := h.SetUp()
		volDir := dir + "/foo/volumes/http/test"
		if test.code != http.StatusOK {
			if err == nil {
				t.Errorf("unexpected non-error")
			}
			_, err := ioutil.ReadDir(volDir)
			if err == nil {
				t.Errorf("unexpected non-error opening %s", volDir)
			}
			continue
		}
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		files, err := ioutil.ReadDir(volDir)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(files) != len(test.files) {
			t.Errorf("unexpected files. expected: %d, saw %d", len(test.files), len(files))
		}
		fileSet := map[string]string{}
		for _, file := range test.files {
			fileSet[file.fileName] = file.contents
		}
		for _, f := range files {
			contents, found := fileSet[f.Name()]
			if !found {
				t.Errorf("unexpected file: %#v", f.Name())
			}
			file, err := os.Open(path.Join(volDir, f.Name()))
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			fileContents, err := ioutil.ReadAll(file)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if string(fileContents) != contents {
				t.Errorf("expected %s, saw %s", contents, fileContents)
			}
		}
		err = h.TearDown()
		if err != nil {
			t.Errorf("unexpeceted error: %v")
		}
	}
}
