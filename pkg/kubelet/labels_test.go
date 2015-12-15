/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package kubelet

import (
	"github.com/stretchr/testify/assert"
	"io/ioutil"
	"k8s.io/kubernetes/pkg/api"
	"os"
	"path"
	"testing"
)

// TestNewNodeLabelSet tests that a LabelSet can be instantiated, or the appropriate error is returned
func TestNewLabelSet(t *testing.T) {
	// Create label dir
	ldir, err := ioutil.TempDir(os.TempDir(), "labels_test_label_dir")
	assert.NoError(t, err)
	defer os.RemoveAll(ldir)

	// Create checkpoint file
	cdir, err := ioutil.TempDir(os.TempDir(), "labels_test_checkoutpoint_dir")
	assert.NoError(t, err)
	defer os.RemoveAll(cdir)
	writeLabelFile(t, cdir, "labelcheckpoint", `{
		"kubernetes.io/key1": "value1"
	}`)
	cpath := path.Join(cdir, "labelcheckpoint")

	// Error case - checkpoint without labeldir
	ls, err := NewLabelSet("", cpath)
	assert.EqualError(t, err, "must specify checkpointFile and labelDirectory together.")
	assert.Nil(t, ls)

	// Error case - labeldir without checkpoint
	ls, err = NewLabelSet(ldir, "")
	assert.EqualError(t, err, "must specify checkpointFile and labelDirectory together.")
	assert.Nil(t, ls)

	// Error case - invalid labeldir
	ls, err = NewLabelSet("/does/not/exist", cpath)
	assert.EqualError(t, err, "open /does/not/exist: no such file or directory")
	assert.Nil(t, ls)

	// Error case - invalid checkpoint
	ls, err = NewLabelSet(ldir, "/does/not/exist.json")
	assert.EqualError(t, err, "cannot create checkpointFile /does/not/exist.json")
	assert.Nil(t, ls)

	// Success - no checkpoint or labeldir
	ls, err = NewLabelSet("", "")
	assert.NoError(t, err)
	assert.NotNil(t, ls)

	// Success - valid checkpoint and labeldir
	ls, err = NewLabelSet(ldir, cpath)
	assert.NoError(t, err)
	assert.NotNil(t, ls)
}

func TestNewLabelSet_CreateCheckpoint(t *testing.T) {
	// Create label dir
	ldir, err := ioutil.TempDir(os.TempDir(), "labels_test_label_dir")
	assert.NoError(t, err)
	defer os.RemoveAll(ldir)

	// Create checkpoint dir
	cdir, err := ioutil.TempDir(os.TempDir(), "labels_test_checkoutpoint_dir")
	assert.NoError(t, err)
	defer os.RemoveAll(cdir)
	cpath := path.Join(cdir, "labelcheckpoint")

	// Check that the file has been created
	ls, err := NewLabelSet(ldir, cpath)
	assert.NoError(t, err)
	assert.NotNil(t, ls)
	_, err = os.Stat(cpath)
	assert.NoError(t, err)
}

// TestWriteNodeLabels tests that node labels read from the directory are written to the map and
// labels read from the checkpoint are removed from the map
func TestWriteNodeLabels(t *testing.T) {
	// Create label dir
	ldir, err := ioutil.TempDir(os.TempDir(), "labels_test_label_dir")
	assert.NoError(t, err)
	defer os.RemoveAll(ldir)
	writeLabelFile(t, ldir, "label1.json", `{
		"addme": "value1"
	}`)
	writeLabelFile(t, ldir, "label2.json", `{
		"keepme": "value2",
		"updateme": "value3"
	}`)

	// Create checkpoint file
	cdir, err := ioutil.TempDir(os.TempDir(), "labels_test_checkoutpoint_dir")
	assert.NoError(t, err)
	defer os.RemoveAll(cdir)
	writeLabelFile(t, cdir, "labelcheckpoint", `{
		"keepme": "",
		"updateme": "",
		"removeme": ""
	}`)
	cpath := path.Join(cdir, "labelcheckpoint")

	// Create the instance
	ls, err := NewLabelSet(ldir, cpath)
	assert.NoError(t, err)
	assert.NotNil(t, ls)

	// Test that the labels are added / removed from the map
	node := &api.Node{
		ObjectMeta: api.ObjectMeta{
			Labels: map[string]string{
				"keepme":   "value2",
				"updateme": "value2",
				"removeme": "value4",
			},
		},
	}

	ls.WriteLabelsToNodeMeta(node)
	expected := map[string]string{
		"addme":    "value1",
		"keepme":   "value2",
		"updateme": "value3",
	}
	assert.Equal(t, expected, node.ObjectMeta.Labels)
}

// TestAppendCheckpoint tests that both the new directory and checkpoint labels are written to the checkpoint file
func TestAppendCheckpoint(t *testing.T) {
	ldir, err := ioutil.TempDir(os.TempDir(), "labels_test_label_dir")
	assert.NoError(t, err)
	defer os.RemoveAll(ldir)

	// Read .json files
	writeLabelFile(t, ldir, "label1.json", `{
		"addme": "value1"
	}`)

	// Read .yaml files
	writeLabelFile(t, ldir, "label2.yaml", "keepme: value2\nupdateme: value3")

	// Ignore .txt files
	writeLabelFile(t, ldir, "label2.txt", `{
		"ignoreme": "value5",
	}`)

	// Create checkpoint file
	cdir, err := ioutil.TempDir(os.TempDir(), "labels_test_checkoutpoint_dir")
	assert.NoError(t, err)
	defer os.RemoveAll(cdir)
	writeLabelFile(t, cdir, "labelcheckpoint", `{
		"keepme": "",
		"updateme": "",
		"removeme": ""
	}`)
	cpath := path.Join(cdir, "labelcheckpoint")

	// Create the instance
	ls, err := NewLabelSet(ldir, cpath)
	assert.NoError(t, err)
	assert.NotNil(t, ls)
	expected := map[string]string{
		"updateme": "",
		"keepme":   "",
		"removeme": "",
	}
	assert.Equal(t, expected, ls.nodeLabelsToClear)

	// Write the new and old labels to the checkpoint file and verify we see them when creating a new LabelSet
	assert.NoError(t, ls.AppendLabelCheckpoint())
	ls, err = NewLabelSet(ldir, cpath)
	assert.NoError(t, err)
	assert.NotNil(t, ls)
	expected = map[string]string{
		"addme":    "",
		"updateme": "",
		"keepme":   "",
		"removeme": "",
	}
	assert.Equal(t, expected, ls.nodeLabelsToClear)

	// Make sure the write is only done once and no futher updates are required
	os.Remove(cpath)
	assert.NoError(t, ls.AppendLabelCheckpoint())
	_, err = os.Stat(cpath)
	assert.EqualError(t, err, "stat "+cpath+": no such file or directory")
}

// TestWriteCheckpoint tests that only the new directory labels are written to the checkpoint file
func TestWriteCheckpoint(t *testing.T) {
	ldir, err := ioutil.TempDir(os.TempDir(), "labels_test_label_dir")
	assert.NoError(t, err)
	defer os.RemoveAll(ldir)
	writeLabelFile(t, ldir, "label1.json", `{
		"addme": "value1"
	}`)
	writeLabelFile(t, ldir, "label2.json", `{
		"keepme": "value2",
		"updateme": "value3"
	}`)

	// Create checkpoint file
	cdir, err := ioutil.TempDir(os.TempDir(), "labels_test_checkoutpoint_dir")
	assert.NoError(t, err)
	defer os.RemoveAll(cdir)
	writeLabelFile(t, cdir, "labelcheckpoint", `{
		"keepme": "",
		"updateme": "",
		"removeme": ""
	}`)
	cpath := path.Join(cdir, "labelcheckpoint")

	// Create the instance
	ls, err := NewLabelSet(ldir, cpath)
	assert.NoError(t, err)
	assert.NotNil(t, ls)
	expected := map[string]string{
		"updateme": "",
		"keepme":   "",
		"removeme": "",
	}
	assert.Equal(t, expected, ls.nodeLabelsToClear)

	// Write the new labels to the checkpoint file and verify we see them when creating a new LabelSet
	assert.NoError(t, ls.WriteLabelCheckpoint())
	ls, err = NewLabelSet(ldir, cpath)
	assert.NoError(t, err)
	assert.NotNil(t, ls)
	expected = map[string]string{
		"addme":    "",
		"keepme":   "",
		"updateme": "",
	}
	assert.Equal(t, expected, ls.nodeLabelsToClear)

	// Make sure the write is only done once and no further updates are required
	os.Remove(cpath)
	assert.NoError(t, ls.WriteLabelCheckpoint())
	_, err = os.Stat(cpath)
	assert.EqualError(t, err, "stat "+cpath+": no such file or directory")
}

// writeLabelFile writes a json string to a file and verifies no error occurred
func writeLabelFile(t *testing.T, tempDir string, name string, json string) {
	err := ioutil.WriteFile(path.Join(tempDir, name), []byte(json), 0664)
	assert.NoError(t, err)
}
