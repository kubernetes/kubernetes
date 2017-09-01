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

package store

import (
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint"
	utilfiles "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/files"
	utilfs "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/filesystem"
	utiltest "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/test"
)

const testCheckpointsDir = "/test-checkpoints-dir"

func newInitializedFakeFsStore() (*fsStore, error) {
	fs := utilfs.NewFakeFs()
	store := NewFsStore(fs, testCheckpointsDir)
	if err := store.Initialize(); err != nil {
		return nil, err
	}
	return store.(*fsStore), nil
}

func TestFsStoreInitialize(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("fsStore.Initialize() failed with error: %v", err)
	}

	// check that testCheckpointsDir exists
	_, err = store.fs.Stat(testCheckpointsDir)
	if err != nil {
		t.Fatalf("expect %q to exist, but stat failed with error: %v", testCheckpointsDir, err)
	}

	// check that testCheckpointsDir contains the curFile
	curPath := filepath.Join(testCheckpointsDir, curFile)
	_, err = store.fs.Stat(curPath)
	if err != nil {
		t.Fatalf("expect %q to exist, but stat failed with error: %v", curPath, err)
	}

	// check that testCheckpointsDir contains the lkgFile
	lkgPath := filepath.Join(testCheckpointsDir, lkgFile)
	_, err = store.fs.Stat(lkgPath)
	if err != nil {
		t.Fatalf("expect %q to exist, but stat failed with error: %v", lkgPath, err)
	}
}

func TestFsStoreExists(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("failed to construct a store, error: %v", err)
	}

	// create a checkpoint file; this is enough for an exists check
	cpt, err := checkpoint.NewConfigMapCheckpoint(&apiv1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{UID: "uid"},
	})
	if err != nil {
		t.Fatalf("could not construct checkpoint, error: %v", err)
	}
	saveTestCheckpointFile(t, store.fs, cpt)

	cases := []struct {
		desc   string
		uid    string // the uid to test
		expect bool
		err    string
	}{
		{"exists", "uid", true, ""},
		{"does not exist", "bogus-uid", false, ""},
	}

	for _, c := range cases {
		ok, err := store.Exists(c.uid)
		if utiltest.SkipRest(t, c.desc, err, c.err) {
			continue
		}
		if c.expect != ok {
			t.Errorf("case %q, expect %t but got %t", c.desc, c.expect, ok)
		}
	}
}

func TestFsStoreSave(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("failed to construct a store, error: %v", err)
	}

	cpt, err := checkpoint.NewConfigMapCheckpoint(&apiv1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{UID: "uid"},
	})
	if err != nil {
		t.Fatalf("could not construct checkpoint, error: %v", err)
	}

	// save the checkpoint
	err = store.Save(cpt)
	if err != nil {
		t.Fatalf("unable to save checkpoint, error: %v", err)
	}

	// expect the saved checkpoint file to match the encoding of the checkpoint
	data, err := cpt.Encode()
	if err != nil {
		t.Fatalf("unable to encode the checkpoint, error: %v", err)
	}
	expect := string(data)

	data = readTestCheckpointFile(t, store.fs, cpt.UID())
	cptFile := string(data)

	if expect != cptFile {
		t.Errorf("expect %q but got %q", expect, cptFile)
	}
}

func TestFsStoreLoad(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("failed to construct a store, error: %v", err)
	}

	const uid = "uid"
	cpt, err := checkpoint.NewConfigMapCheckpoint(&apiv1.ConfigMap{ObjectMeta: metav1.ObjectMeta{UID: types.UID(uid)}})
	if err != nil {
		t.Fatalf("unable to construct checkpoint, error: %v", err)
	}

	cases := []struct {
		desc    string
		loadUID string
		cpt     checkpoint.Checkpoint
		err     string
	}{
		{"checkpoint exists", uid, cpt, ""},
		{"checkpoint does not exist", "bogus-uid", nil, "failed to read"},
	}
	for _, c := range cases {
		if c.cpt != nil {
			saveTestCheckpointFile(t, store.fs, c.cpt)
		}
		cpt, err := store.Load(c.loadUID)
		if utiltest.SkipRest(t, c.desc, err, c.err) {
			continue
		}
		if !checkpoint.EqualCheckpoints(c.cpt, cpt) {
			t.Errorf("case %q, expect %q but got %q", c.desc, spew.Sdump(c.cpt), spew.Sdump(cpt))
		}
	}
}

func TestFsStoreRoundTrip(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("failed to construct a store, error: %v", err)
	}
	const uid = "uid"
	cpt, err := checkpoint.NewConfigMapCheckpoint(&apiv1.ConfigMap{ObjectMeta: metav1.ObjectMeta{UID: types.UID(uid)}})
	if err != nil {
		t.Fatalf("unable to construct checkpoint, error: %v", err)
	}
	err = store.Save(cpt)
	if err != nil {
		t.Fatalf("unable to save checkpoint, error: %v", err)
	}
	cptAfter, err := store.Load(uid)
	if err != nil {
		t.Fatalf("unable to load checkpoint, error: %v", err)
	}
	if !checkpoint.EqualCheckpoints(cpt, cptAfter) {
		t.Errorf("expect %q but got %q", spew.Sdump(cpt), spew.Sdump(cptAfter))
	}
}

func TestFsStoreCurrentModified(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("failed to construct a store, error: %v", err)
	}

	// create an empty current file, this is good enough for testing
	saveTestSourceFile(t, store.fs, curFile, nil)

	// set the timestamps to the current time, so we can compare to result of store.SetCurrentModified
	now := time.Now()
	err = store.fs.Chtimes(filepath.Join(testCheckpointsDir, curFile), now, now)
	if err != nil {
		t.Fatalf("could not change timestamps, error: %v", err)
	}

	// for now we hope that the system won't truncate the time to a less precise unit,
	// if this test fails on certain systems that may be the reason.
	modTime, err := store.CurrentModified()
	if err != nil {
		t.Fatalf("unable to determine modification time of current config source, error: %v", err)
	}
	if !now.Equal(modTime) {
		t.Errorf("expect %q but got %q", now.Format(time.RFC3339), modTime.Format(time.RFC3339))
	}
}

func TestFsStoreCurrent(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("failed to construct a store, error: %v", err)
	}

	source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{
		ConfigMapRef: &apiv1.ObjectReference{Name: "name", Namespace: "namespace", UID: "uid"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cases := []struct {
		desc   string
		expect checkpoint.RemoteConfigSource
		err    string
	}{
		{"default source", nil, ""},
		{"non-default source", source, ""},
	}
	for _, c := range cases {
		// save the last known good source
		saveTestSourceFile(t, store.fs, curFile, c.expect)

		// load last-known-good and compare to expected result
		source, err := store.Current()
		if utiltest.SkipRest(t, c.desc, err, c.err) {
			continue
		}
		if !checkpoint.EqualRemoteConfigSources(c.expect, source) {
			t.Errorf("case %q, expect %q but got %q", spew.Sdump(c.expect), spew.Sdump(c.expect), spew.Sdump(source))
		}
	}
}

func TestFsStoreLastKnownGood(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("failed to construct a store, error: %v", err)
	}

	source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{
		ConfigMapRef: &apiv1.ObjectReference{Name: "name", Namespace: "namespace", UID: "uid"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cases := []struct {
		desc   string
		expect checkpoint.RemoteConfigSource
		err    string
	}{
		{"default source", nil, ""},
		{"non-default source", source, ""},
	}
	for _, c := range cases {
		// save the last known good source
		saveTestSourceFile(t, store.fs, lkgFile, c.expect)

		// load last-known-good and compare to expected result
		source, err := store.LastKnownGood()
		if utiltest.SkipRest(t, c.desc, err, c.err) {
			continue
		}
		if !checkpoint.EqualRemoteConfigSources(c.expect, source) {
			t.Errorf("case %q, expect %q but got %q", spew.Sdump(c.expect), spew.Sdump(c.expect), spew.Sdump(source))
		}
	}
}

func TestFsStoreSetCurrent(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("failed to construct a store, error: %v", err)
	}

	const uid = "uid"
	expect := fmt.Sprintf(`{"kind":"NodeConfigSource","apiVersion":"v1","configMapRef":{"namespace":"namespace","name":"name","uid":"%s"}}%s`, uid, "\n")
	source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{
		Name: "name", Namespace: "namespace", UID: types.UID(uid)}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// save the current source
	if err := store.SetCurrent(source); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// check that the source saved as we would expect
	data := readTestSourceFile(t, store.fs, curFile)
	if expect != string(data) {
		t.Errorf("expect current source file to contain %q, but got %q", expect, string(data))
	}
}

func TestFsStoreSetCurrentUpdated(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("failed to construct a store, error: %v", err)
	}

	cases := []struct {
		current       string
		newCurrent    string
		expectUpdated bool
		err           string
	}{
		{"", "", false, ""},
		{"uid", "", true, ""},
		{"", "uid", true, ""},
		{"uid", "uid", false, ""},
		{"uid", "other-uid", true, ""},
		{"other-uid", "uid", true, ""},
		{"other-uid", "other-uid", false, ""},
	}

	for _, c := range cases {
		// construct current source
		var source checkpoint.RemoteConfigSource
		expectSource := ""
		if len(c.current) > 0 {
			expectSource = fmt.Sprintf(`{"kind":"NodeConfigSource","apiVersion":"v1","configMapRef":{"namespace":"namespace","name":"name","uid":"%s"}}%s`, c.current, "\n")
			source, _, err = checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{
				Name: "name", Namespace: "namespace", UID: types.UID(c.current)}})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		}

		// construct new source
		var newSource checkpoint.RemoteConfigSource
		expectNewSource := ""
		if len(c.newCurrent) > 0 {
			expectNewSource = fmt.Sprintf(`{"kind":"NodeConfigSource","apiVersion":"v1","configMapRef":{"namespace":"namespace","name":"new-name","uid":"%s"}}%s`, c.newCurrent, "\n")
			newSource, _, err = checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{
				Name: "new-name", Namespace: "namespace", UID: types.UID(c.newCurrent)}})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		}

		// set the initial current
		if err := store.SetCurrent(source); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// update to the new current
		updated, err := store.SetCurrentUpdated(newSource)
		if utiltest.SkipRest(t, fmt.Sprintf("%q -> %q", c.current, c.newCurrent), err, c.err) {
			continue
		}

		// check that SetCurrentUpdated correctly reports whether the current checkpoint changed
		if c.expectUpdated != updated {
			t.Errorf("case %q -> %q, expect %v but got %v", c.current, c.newCurrent, c.expectUpdated, updated)
		}

		// check that curFile is saved by SetCurrentUpdated as we expect
		data := readTestSourceFile(t, store.fs, curFile)
		if c.current == c.newCurrent {
			// same UID should leave file unchanged
			if expectSource != string(data) {
				t.Errorf("case %q -> %q, expect current source file to contain %q, but got %q", c.current, c.newCurrent, expectSource, string(data))
			}
		} else if expectNewSource != string(data) {
			// otherwise expect the file to change
			t.Errorf("case %q -> %q, expect current source file to contain %q, but got %q", c.current, c.newCurrent, expectNewSource, string(data))
		}
	}

}

func TestFsStoreSetLastKnownGood(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("failed to construct a store, error: %v", err)
	}

	const uid = "uid"
	expect := fmt.Sprintf(`{"kind":"NodeConfigSource","apiVersion":"v1","configMapRef":{"namespace":"namespace","name":"name","uid":"%s"}}%s`, uid, "\n")
	source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{
		Name: "name", Namespace: "namespace", UID: types.UID(uid)}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// save the last known good source
	if err := store.SetLastKnownGood(source); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// check that the source saved as we would expect
	data := readTestSourceFile(t, store.fs, lkgFile)
	if expect != string(data) {
		t.Errorf("expect last-known-good source file to contain %q, but got %q", expect, string(data))
	}
}

func TestFsStoreReset(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("failed to construct a store, error: %v", err)
	}

	source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{Name: "name", Namespace: "namespace", UID: "uid"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	otherSource, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{Name: "other-name", Namespace: "namespace", UID: "other-uid"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	cases := []struct {
		desc          string
		current       checkpoint.RemoteConfigSource
		lastKnownGood checkpoint.RemoteConfigSource
		updated       bool
	}{
		{"nil -> nil", nil, nil, false},
		{"source -> nil", source, nil, true},
		{"nil -> source", nil, source, false},
		{"source -> source", source, source, true},
		{"source -> otherSource", source, otherSource, true},
		{"otherSource -> source", otherSource, source, true},
	}
	for _, c := range cases {
		// manually save the sources to their respective files
		saveTestSourceFile(t, store.fs, curFile, c.current)
		saveTestSourceFile(t, store.fs, lkgFile, c.lastKnownGood)

		// reset
		updated, err := store.Reset()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// make sure the files were emptied
		if size := testSourceFileSize(t, store.fs, curFile); size > 0 {
			t.Errorf("case %q, expect source file %q to be empty but got %d bytes", c.desc, curFile, size)
		}
		if size := testSourceFileSize(t, store.fs, lkgFile); size > 0 {
			t.Errorf("case %q, expect source file %q to be empty but got %d bytes", c.desc, lkgFile, size)
		}

		// make sure Current() and LastKnownGood() both return nil
		current, err := store.Current()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		lastKnownGood, err := store.LastKnownGood()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if current != nil || lastKnownGood != nil {
			t.Errorf("case %q, expect nil for current and last-known-good checkpoints, but still have %q and %q, respectively",
				c.desc, current, lastKnownGood)
		}
		if c.updated != updated {
			t.Errorf("case %q, expect reset to return %t, but got %t", c.desc, c.updated, updated)
		}
	}
}

func TestFsStoreSourceFromFile(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("failed to construct a store, error: %v", err)
	}

	source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{
		ConfigMapRef: &apiv1.ObjectReference{Name: "name", Namespace: "namespace", UID: "uid"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cases := []struct {
		desc   string
		expect checkpoint.RemoteConfigSource
		err    string
	}{
		{"default source", nil, ""},
		{"non-default source", source, ""},
	}

	const name = "some-source-file"
	for _, c := range cases {
		saveTestSourceFile(t, store.fs, name, c.expect)
		source, err := store.sourceFromFile(name)
		if utiltest.SkipRest(t, c.desc, err, c.err) {
			continue
		}
		if !checkpoint.EqualRemoteConfigSources(c.expect, source) {
			t.Errorf("case %q, expect %q but got %q", spew.Sdump(c.expect), spew.Sdump(c.expect), spew.Sdump(source))
		}
	}
}

func TestFsStoreSetSourceFile(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("failed to construct a store, error: %v", err)
	}

	source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{Name: "name", Namespace: "namespace", UID: "uid"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cases := []struct {
		source checkpoint.RemoteConfigSource
	}{
		{nil},
		{source},
	}

	const name = "some-source-file"
	for _, c := range cases {
		// set the source file
		err := store.setSourceFile(name, c.source)
		if err != nil {
			t.Fatalf("unable to set source file, error: %v", err)
		}
		// read back the file
		data := readTestSourceFile(t, store.fs, name)
		str := string(data)

		if c.source != nil {
			// expect the contents to match the encoding of the source
			data, err := c.source.Encode()
			expect := string(data)
			if err != nil {
				t.Fatalf("couldn't encode source, error: %v", err)
			}
			if expect != str {
				t.Errorf("case %q, expect %q but got %q", spew.Sdump(c.source), expect, str)
			}
		} else {
			// expect empty file
			expect := ""
			if expect != str {
				t.Errorf("case %q, expect %q but got %q", spew.Sdump(c.source), expect, str)
			}
		}
	}
}

func readTestCheckpointFile(t *testing.T, fs utilfs.Filesystem, uid string) []byte {
	data, err := fs.ReadFile(filepath.Join(testCheckpointsDir, uid))
	if err != nil {
		t.Fatalf("unable to read test checkpoint file, error: %v", err)
	}
	return data
}

func saveTestCheckpointFile(t *testing.T, fs utilfs.Filesystem, cpt checkpoint.Checkpoint) {
	data, err := cpt.Encode()
	if err != nil {
		t.Fatalf("unable to encode test checkpoint, error: %v", err)
	}
	fmt.Println(cpt.UID())
	err = utilfiles.ReplaceFile(fs, filepath.Join(testCheckpointsDir, cpt.UID()), data)
	if err != nil {
		t.Fatalf("unable to save test checkpoint file, error: %v", err)
	}
}

func readTestSourceFile(t *testing.T, fs utilfs.Filesystem, relPath string) []byte {
	data, err := fs.ReadFile(filepath.Join(testCheckpointsDir, relPath))
	if err != nil {
		t.Fatalf("unable to read test source file, error: %v", err)
	}
	return data
}

func saveTestSourceFile(t *testing.T, fs utilfs.Filesystem, relPath string, source checkpoint.RemoteConfigSource) {
	if source != nil {
		data, err := source.Encode()
		if err != nil {
			t.Fatalf("unable to save test source file, error: %v", err)
		}
		err = utilfiles.ReplaceFile(fs, filepath.Join(testCheckpointsDir, relPath), data)
		if err != nil {
			t.Fatalf("unable to save test source file, error: %v", err)
		}
	} else {
		err := utilfiles.ReplaceFile(fs, filepath.Join(testCheckpointsDir, relPath), []byte{})
		if err != nil {
			t.Fatalf("unable to save test source file, error: %v", err)
		}
	}
}

func testSourceFileSize(t *testing.T, fs utilfs.Filesystem, relPath string) int64 {
	info, err := fs.Stat(filepath.Join(testCheckpointsDir, relPath))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	return info.Size()
}
