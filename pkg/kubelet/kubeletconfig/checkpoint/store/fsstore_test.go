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
	"io/ioutil"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/scheme"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint"
	utilcodec "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/codec"
	utilfiles "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/files"
	utiltest "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/test"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

var testdir string

func init() {
	tmp, err := ioutil.TempDir("", "fsstore-test")
	if err != nil {
		panic(err)
	}
	testdir = tmp
}

func newInitializedFakeFsStore() (*fsStore, error) {
	// Test with the default filesystem, the fake filesystem has an issue caused by afero: https://github.com/spf13/afero/issues/141
	// The default filesystem also behaves more like production, so we should probably not mock the filesystem for unit tests.
	fs := utilfs.DefaultFs{}

	tmpdir, err := fs.TempDir(testdir, "store-")
	if err != nil {
		return nil, err
	}

	store := NewFsStore(fs, tmpdir)
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

	// check that store.dir exists
	if _, err := store.fs.Stat(store.dir); err != nil {
		t.Fatalf("expect %q to exist, but stat failed with error: %v", store.dir, err)
	}

	// check that meta dir exists
	if _, err := store.fs.Stat(store.metaPath("")); err != nil {
		t.Fatalf("expect %q to exist, but stat failed with error: %v", store.metaPath(""), err)
	}

	// check that checkpoints dir exists
	if _, err := store.fs.Stat(store.checkpointPath("")); err != nil {
		t.Fatalf("expect %q to exist, but stat failed with error: %v", store.checkpointPath(""), err)
	}

	// check that currentFile exists
	if _, err := store.fs.Stat(store.metaPath(currentFile)); err != nil {
		t.Fatalf("expect %q to exist, but stat failed with error: %v", store.metaPath(currentFile), err)
	}

	// check that lastKnownGoodFile exists
	if _, err := store.fs.Stat(store.metaPath(lastKnownGoodFile)); err != nil {
		t.Fatalf("expect %q to exist, but stat failed with error: %v", store.metaPath(lastKnownGoodFile), err)
	}
}

func TestFsStoreExists(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("error constructing store: %v", err)
	}

	// checkpoint a payload
	const uid = "uid"
	p, err := checkpoint.NewConfigMapPayload(&apiv1.ConfigMap{ObjectMeta: metav1.ObjectMeta{UID: uid}})
	if err != nil {
		t.Fatalf("could not construct checkpoint, error: %v", err)
	}
	store.Save(p)

	cases := []struct {
		desc   string
		uid    types.UID
		expect bool
		err    string
	}{
		{"exists", uid, true, ""},
		{"does not exist", "bogus-uid", false, ""},
	}

	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{
				ConfigMap: &apiv1.ConfigMapNodeConfigSource{
					Name:             "name",
					Namespace:        "namespace",
					UID:              c.uid,
					KubeletConfigKey: "kubelet",
				}})
			if err != nil {
				t.Fatalf("error constructing remote config source: %v", err)
			}
			ok, err := store.Exists(source)
			utiltest.ExpectError(t, err, c.err)
			if err != nil {
				return
			}
			if c.expect != ok {
				t.Errorf("expect %t but got %t", c.expect, ok)
			}
		})
	}
}

func TestFsStoreSave(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("error constructing store: %v", err)
	}

	nameTooLong := func() string {
		s := ""
		for i := 0; i < 256; i++ {
			s += "a"
		}
		return s
	}()

	cases := []struct {
		desc  string
		files map[string]string
		err   string
	}{
		{"valid payload", map[string]string{"foo": "foocontent", "bar": "barcontent"}, ""},
		{"empty key name", map[string]string{"": "foocontent"}, "must not be empty"},
		{"key name is not a base file name (foo/bar)", map[string]string{"foo/bar": "foocontent"}, "only base names are allowed"},
		{"key name is not a base file name (/foo)", map[string]string{"/bar": "foocontent"}, "only base names are allowed"},
		{"used .", map[string]string{".": "foocontent"}, "may not be '.' or '..'"},
		{"used ..", map[string]string{"..": "foocontent"}, "may not be '.' or '..'"},
		{"length violation", map[string]string{nameTooLong: "foocontent"}, "must be less than 255 characters"},
	}

	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			// construct the payload
			p, err := checkpoint.NewConfigMapPayload(&apiv1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{UID: "uid"},
				Data:       c.files,
			})
			if err != nil {
				t.Fatalf("error constructing payload: %v", err)
			}
			// save the payload
			err = store.Save(p)
			utiltest.ExpectError(t, err, c.err)
			if err != nil {
				return
			}
			// read the saved checkpoint
			m, err := mapFromCheckpoint(store, p.UID())
			if err != nil {
				t.Fatalf("error loading checkpoint to map: %v", err)
			}
			// compare our expectation to what got saved
			expect := p.Files()
			if !reflect.DeepEqual(expect, m) {
				t.Errorf("expect %v, but got %v", expect, m)
			}
		})
	}
}

func TestFsStoreLoad(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("error constructing store: %v", err)
	}
	// encode a kubelet configuration that has all defaults set
	expect, err := newKubeletConfiguration()
	if err != nil {
		t.Fatalf("error constructing KubeletConfiguration: %v", err)
	}
	data, err := utilcodec.EncodeKubeletConfig(expect, v1beta1.SchemeGroupVersion)
	if err != nil {
		t.Fatalf("error encoding KubeletConfiguration: %v", err)
	}
	// construct a payload that contains the kubeletconfig
	const (
		uid        = "uid"
		kubeletKey = "kubelet"
	)
	p, err := checkpoint.NewConfigMapPayload(&apiv1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{UID: types.UID(uid)},
		Data: map[string]string{
			kubeletKey: string(data),
		},
	})
	if err != nil {
		t.Fatalf("error constructing payload: %v", err)
	}
	// save the payload
	err = store.Save(p)
	if err != nil {
		t.Fatalf("error saving payload: %v", err)
	}

	cases := []struct {
		desc string
		uid  types.UID
		err  string
	}{
		{"checkpoint exists", uid, ""},
		{"checkpoint does not exist", "bogus-uid", "no checkpoint for source"},
	}
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{
				ConfigMap: &apiv1.ConfigMapNodeConfigSource{
					Name:             "name",
					Namespace:        "namespace",
					UID:              c.uid,
					KubeletConfigKey: kubeletKey,
				}})
			if err != nil {
				t.Fatalf("error constructing remote config source: %v", err)
			}
			loaded, err := store.Load(source)
			utiltest.ExpectError(t, err, c.err)
			if err != nil {
				return
			}
			if !reflect.DeepEqual(expect, loaded) {
				t.Errorf("expect %#v, but got %#v", expect, loaded)
			}
		})
	}
}

func TestFsStoreCurrentModified(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("error constructing store: %v", err)
	}

	// create an empty current file, this is good enough for testing
	saveTestSourceFile(t, store, currentFile, nil)

	// set the timestamps to the current time, so we can compare to result of store.CurrentModified
	now := time.Now()
	err = store.fs.Chtimes(store.metaPath(currentFile), now, now)
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
		t.Fatalf("error constructing store: %v", err)
	}

	source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{
		ConfigMap: &apiv1.ConfigMapNodeConfigSource{
			Name:             "name",
			Namespace:        "namespace",
			UID:              "uid",
			KubeletConfigKey: "kubelet",
		}})
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
		t.Run(c.desc, func(t *testing.T) {
			// save the last known good source
			saveTestSourceFile(t, store, currentFile, c.expect)

			// load last-known-good and compare to expected result
			source, err := store.Current()
			utiltest.ExpectError(t, err, c.err)
			if err != nil {
				return
			}
			if !checkpoint.EqualRemoteConfigSources(c.expect, source) {
				t.Errorf("case %q, expect %q but got %q", spew.Sdump(c.expect), spew.Sdump(c.expect), spew.Sdump(source))
			}
		})
	}
}

func TestFsStoreLastKnownGood(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("error constructing store: %v", err)
	}

	source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{
		ConfigMap: &apiv1.ConfigMapNodeConfigSource{
			Name:             "name",
			Namespace:        "namespace",
			UID:              "uid",
			KubeletConfigKey: "kubelet",
		}})
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
		t.Run(c.desc, func(t *testing.T) {
			// save the last known good source
			saveTestSourceFile(t, store, lastKnownGoodFile, c.expect)

			// load last-known-good and compare to expected result
			source, err := store.LastKnownGood()
			utiltest.ExpectError(t, err, c.err)
			if err != nil {
				return
			}
			if !checkpoint.EqualRemoteConfigSources(c.expect, source) {
				t.Errorf("case %q, expect %q but got %q", spew.Sdump(c.expect), spew.Sdump(c.expect), spew.Sdump(source))
			}
		})
	}
}

func TestFsStoreSetCurrent(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("error constructing store: %v", err)
	}

	const uid = "uid"
	expect := fmt.Sprintf(`apiVersion: kubelet.config.k8s.io/v1beta1
kind: SerializedNodeConfigSource
source:
  configMap:
    kubeletConfigKey: kubelet
    name: name
    namespace: namespace
    uid: %s
`, uid)
	source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
		Name:             "name",
		Namespace:        "namespace",
		UID:              types.UID(uid),
		KubeletConfigKey: "kubelet",
	}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// save the current source
	if err := store.SetCurrent(source); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// check that the source saved as we would expect
	data := readTestSourceFile(t, store, currentFile)
	if expect != string(data) {
		t.Errorf("expect current source file to contain %q, but got %q", expect, string(data))
	}
}

func TestFsStoreSetLastKnownGood(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("error constructing store: %v", err)
	}

	const uid = "uid"
	expect := fmt.Sprintf(`apiVersion: kubelet.config.k8s.io/v1beta1
kind: SerializedNodeConfigSource
source:
  configMap:
    kubeletConfigKey: kubelet
    name: name
    namespace: namespace
    uid: %s
`, uid)
	source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
		Name:             "name",
		Namespace:        "namespace",
		UID:              types.UID(uid),
		KubeletConfigKey: "kubelet",
	}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// save the last known good source
	if err := store.SetLastKnownGood(source); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// check that the source saved as we would expect
	data := readTestSourceFile(t, store, lastKnownGoodFile)
	if expect != string(data) {
		t.Errorf("expect last-known-good source file to contain %q, but got %q", expect, string(data))
	}
}

func TestFsStoreReset(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("error constructing store: %v", err)
	}

	source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
		Name:             "name",
		Namespace:        "namespace",
		UID:              "uid",
		KubeletConfigKey: "kubelet",
	}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	otherSource, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
		Name:             "other-name",
		Namespace:        "namespace",
		UID:              "other-uid",
		KubeletConfigKey: "kubelet",
	}})
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
		t.Run(c.desc, func(t *testing.T) {
			// manually save the sources to their respective files
			saveTestSourceFile(t, store, currentFile, c.current)
			saveTestSourceFile(t, store, lastKnownGoodFile, c.lastKnownGood)

			// reset
			updated, err := store.Reset()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// make sure the files were emptied
			if size := testSourceFileSize(t, store, currentFile); size > 0 {
				t.Errorf("case %q, expect source file %q to be empty but got %d bytes", c.desc, currentFile, size)
			}
			if size := testSourceFileSize(t, store, lastKnownGoodFile); size > 0 {
				t.Errorf("case %q, expect source file %q to be empty but got %d bytes", c.desc, lastKnownGoodFile, size)
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
		})
	}
}

func TestFsStoreReadRemoteConfigSource(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("error constructing store: %v", err)
	}

	source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{
		ConfigMap: &apiv1.ConfigMapNodeConfigSource{
			Name:             "name",
			Namespace:        "namespace",
			UID:              "uid",
			KubeletConfigKey: "kubelet",
		}})
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
		t.Run(c.desc, func(t *testing.T) {
			saveTestSourceFile(t, store, name, c.expect)
			source, err := readRemoteConfigSource(store.fs, store.metaPath(name))
			utiltest.ExpectError(t, err, c.err)
			if err != nil {
				return
			}
			if !checkpoint.EqualRemoteConfigSources(c.expect, source) {
				t.Errorf("case %q, expect %q but got %q", spew.Sdump(c.expect), spew.Sdump(c.expect), spew.Sdump(source))
			}
		})
	}
}

func TestFsStoreWriteRemoteConfigSource(t *testing.T) {
	store, err := newInitializedFakeFsStore()
	if err != nil {
		t.Fatalf("error constructing store: %v", err)
	}

	source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
		Name:             "name",
		Namespace:        "namespace",
		UID:              "uid",
		KubeletConfigKey: "kubelet",
	}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cases := []struct {
		desc   string
		source checkpoint.RemoteConfigSource
	}{
		{"nil source", nil},
		{"non-nil source", source},
	}

	const name = "some-source-file"
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			// set the source file
			err := writeRemoteConfigSource(store.fs, store.metaPath(name), c.source)
			if err != nil {
				t.Fatalf("unable to set source file, error: %v", err)
			}
			// read back the file
			data := readTestSourceFile(t, store, name)
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
		})
	}
}

func mapFromCheckpoint(store *fsStore, uid string) (map[string]string, error) {
	files, err := store.fs.ReadDir(store.checkpointPath(uid))
	if err != nil {
		return nil, err
	}
	m := map[string]string{}
	for _, f := range files {
		// expect no subdirs, only regular files
		if !f.Mode().IsRegular() {
			return nil, fmt.Errorf("expect only regular files in checkpoint dir %q", uid)
		}
		// read the file contents and build the map
		data, err := store.fs.ReadFile(filepath.Join(store.checkpointPath(uid), f.Name()))
		if err != nil {
			return nil, err
		}
		m[f.Name()] = string(data)
	}
	return m, nil
}

func readTestSourceFile(t *testing.T, store *fsStore, relPath string) []byte {
	data, err := store.fs.ReadFile(store.metaPath(relPath))
	if err != nil {
		t.Fatalf("unable to read test source file, error: %v", err)
	}
	return data
}

func saveTestSourceFile(t *testing.T, store *fsStore, relPath string, source checkpoint.RemoteConfigSource) {
	if source != nil {
		data, err := source.Encode()
		if err != nil {
			t.Fatalf("unable to save test source file, error: %v", err)
		}
		err = utilfiles.ReplaceFile(store.fs, store.metaPath(relPath), data)
		if err != nil {
			t.Fatalf("unable to save test source file, error: %v", err)
		}
	} else {
		err := utilfiles.ReplaceFile(store.fs, store.metaPath(relPath), []byte{})
		if err != nil {
			t.Fatalf("unable to save test source file, error: %v", err)
		}
	}
}

func testSourceFileSize(t *testing.T, store *fsStore, relPath string) int64 {
	info, err := store.fs.Stat(store.metaPath(relPath))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	return info.Size()
}

// newKubeletConfiguration will create a new KubeletConfiguration with default values set
func newKubeletConfiguration() (*kubeletconfig.KubeletConfiguration, error) {
	s, _, err := scheme.NewSchemeAndCodecs()
	if err != nil {
		return nil, err
	}
	versioned := &v1beta1.KubeletConfiguration{}
	s.Default(versioned)
	config := &kubeletconfig.KubeletConfiguration{}
	if err := s.Convert(versioned, config, nil); err != nil {
		return nil, err
	}
	return config, nil
}
