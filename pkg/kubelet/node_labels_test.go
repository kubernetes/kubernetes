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

package kubelet_test

import (
	"github.com/stretchr/testify/assert"
	"io/ioutil"
	"k8s.io/kubernetes/pkg/kubelet"
	"os"
	"path"
	"testing"
)

func TestGetNodeLabels_Empty(t *testing.T) {
	nlm := kubelet.NewNodeLabelMap("", "")
	actual := nlm.GetNodeLabels()
	expected := map[string]string{
		"kubernetes.io/hostname": "",
	}
	assert.Equal(t, expected, actual)

	nlm.UpdateNodeLabels()
	actual = nlm.GetNodeLabels()
	assert.Equal(t, expected, actual)
}

func TestGetNodeLabels_TopNameSpace(t *testing.T) {
	name, err := ioutil.TempDir(os.TempDir(), "node_labels_test")
	defer os.RemoveAll(name)
	assert.NoError(t, err)

	writeJsonFile(t, name, "label1.json", `{
		"a.node.kubernetes.io/key1": "value1",
		"b.node.kubernetes.io/key2": "value2"
	}`)

	writeJsonFile(t, name, "label2.json", `{
		"a.node.kubernetes.io/key3": "value3",
		"b.node.kubernetes.io/key4": "value4"
	}`)

	nlm := kubelet.NewNodeLabelMap("hn", name)
	nlm.UpdateNodeLabels()
	actual := nlm.GetNodeLabels()
	expected := map[string]string{
		"kubernetes.io/hostname":    "hn",
		"a.node.kubernetes.io/key1": "value1",
		"b.node.kubernetes.io/key2": "value2",
		"a.node.kubernetes.io/key3": "value3",
		"b.node.kubernetes.io/key4": "value4",
	}
	assert.Equal(t, expected, actual)
}

func TestGetNodeLabels_JsonOnly(t *testing.T) {
	name, err := ioutil.TempDir(os.TempDir(), "node_labels_test")
	defer os.RemoveAll(name)
	assert.NoError(t, err)

	writeJsonFile(t, name, "label1.json", `{
		"a.node.kubernetes.io/key1": "value1",
		"b.node.kubernetes.io/key2": "value2"
	}`)

	writeJsonFile(t, name, "label2.js", `{
		"a.node.kubernetes.io/key3": "value3",
		"b.node.kubernetes.io/key4": "value4"
	}`)

	nlm := kubelet.NewNodeLabelMap("hn", name)
	nlm.UpdateNodeLabels()
	actual := nlm.GetNodeLabels()
	assert.NoError(t, err)
	expected := map[string]string{
		"kubernetes.io/hostname":    "hn",
		"a.node.kubernetes.io/key1": "value1",
		"b.node.kubernetes.io/key2": "value2",
	}
	assert.Equal(t, expected, actual)
}

func TestGetNodeLabels_ExecSh(t *testing.T) {
	name, err := ioutil.TempDir(os.TempDir(), "node_labels_test")
	defer os.RemoveAll(name)
	assert.NoError(t, err)

	sh := `#!/bin/bash
		echo '{'
		echo '"a.node.kubernetes.io/key1": "value1",'
		echo '"b.node.kubernetes.io/key2": "value2"'
		echo '}'
	`
	err = ioutil.WriteFile(path.Join(name, "label1.sh"), []byte(sh), 0555)
	assert.NoError(t, err)

	nlm := kubelet.NewNodeLabelMap("hn", name)
	nlm.UpdateNodeLabels()
	actual := nlm.GetNodeLabels()
	assert.NoError(t, err)
	expected := map[string]string{
		"kubernetes.io/hostname":    "hn",
		"a.node.kubernetes.io/key1": "value1",
		"b.node.kubernetes.io/key2": "value2",
	}
	assert.Equal(t, expected, actual)
}

func TestGetNodeLabels_JsonAndExecSh(t *testing.T) {
	name, err := ioutil.TempDir(os.TempDir(), "node_labels_test")
	defer os.RemoveAll(name)
	assert.NoError(t, err)

	sh := `#!/bin/bash
		echo '{'
		echo '"a.node.kubernetes.io/key1": "value1",'
		echo '"a.node.kubernetes.io/key2": "value2"'
		echo '}'
	`
	err = ioutil.WriteFile(path.Join(name, "label2.sh"), []byte(sh), 0555)
	assert.NoError(t, err)

	sh = `#!/bin/bash
		echo '{'
		echo '"a.node.kubernetes.io/key3": "value1",'
		echo '"a.node.kubernetes.io/key4": "value2"'
		echo '}'
	`
	err = ioutil.WriteFile(path.Join(name, "label1.sh"), []byte(sh), 0555)
	assert.NoError(t, err)

	writeJsonFile(t, name, "label1.json", `{
		"b.node.kubernetes.io/key3": "value3",
		"b.node.kubernetes.io/key4": "value4"
	}`)

	writeJsonFile(t, name, "label2.json", `{
		"b.node.kubernetes.io/key5": "value3",
		"b.node.kubernetes.io/key6": "value4"
	}`)

	nlm := kubelet.NewNodeLabelMap("hn", name)
	nlm.UpdateNodeLabels()
	actual := nlm.GetNodeLabels()
	assert.NoError(t, err)
	expected := map[string]string{
		"kubernetes.io/hostname":    "hn",
		"a.node.kubernetes.io/key1": "value1",
		"a.node.kubernetes.io/key2": "value2",
		"a.node.kubernetes.io/key3": "value1",
		"a.node.kubernetes.io/key4": "value2",
		"b.node.kubernetes.io/key3": "value3",
		"b.node.kubernetes.io/key4": "value4",
		"b.node.kubernetes.io/key5": "value3",
		"b.node.kubernetes.io/key6": "value4",
	}
	assert.Equal(t, expected, actual)
}

func TestGetNodeLabels_DuplicateLabels(t *testing.T) {
	name, err := ioutil.TempDir(os.TempDir(), "node_labels_test")
	defer os.RemoveAll(name)
	assert.NoError(t, err)

	writeJsonFile(t, name, "label1.json", `{
		"a.node.kubernetes.io/key1": "value1",
		"a.node.kubernetes.io/key2": "value2"
	}`)

	writeJsonFile(t, name, "label2.json", `{
		"a.node.kubernetes.io/key3": "value3",
		"a.node.kubernetes.io/key2": "value4"
	}`)

	nlm := kubelet.NewNodeLabelMap("hn", name)
	nlm.UpdateNodeLabels()
	actual := nlm.GetNodeLabels()
	expected := map[string]string{
		"kubernetes.io/hostname": "hn",
	}
	assert.Equal(t, expected, actual)
}

func TestGetNodeLabels_BadNameSpace(t *testing.T) {
	name, err := ioutil.TempDir(os.TempDir(), "node_labels_test")
	defer os.RemoveAll(name)
	assert.NoError(t, err)

	writeJsonFile(t, name, "label1.json", `{
		"node.kubernetes.io/key1": "value1",
		"my.ns/key2": "value1",
		"node.kubernetes.io/key3": "value3"
	}`)

	nlm := kubelet.NewNodeLabelMap("hn", name)
	nlm.UpdateNodeLabels()
	actual := nlm.GetNodeLabels()
	expected := map[string]string{
		"kubernetes.io/hostname": "hn",
	}
	assert.Equal(t, expected, actual)
}

func TestParseJson_BadNameSpace(t *testing.T) {
	nlm := kubelet.NewNodeLabelMap("", "")

	// Verify custom child namespace is ok
	actual := make(map[string]string)
	err := nlm.ParseJson([]byte(`{"a.node.kubernetes.io/a": "value3"}`), actual)
	assert.NoError(t, err)
	assert.Equal(t, map[string]string{"a.node.kubernetes.io/a": "value3"}, actual)

	// Verify default namespace is not ok
	actual = make(map[string]string)
	err = nlm.ParseJson([]byte(`{"node.kubernetes.io/a": "value3"}`), actual)
	assert.Error(t, err)
	assert.Equal(t, make(map[string]string), actual)

	// Verify parent namespace is not allowed
	actual = make(map[string]string)
	err = nlm.ParseJson([]byte(`{"kubernetes.io/key3": "value3"}`), actual)
	assert.Error(t, err)
	assert.Equal(t, make(map[string]string), actual)

	// Verify sibling namespace is not allowed
	actual = make(map[string]string)
	err = nlm.ParseJson([]byte(`{"anode.kubernetes.io/key3": "value3"}`), actual)
	assert.Error(t, err)
	assert.Equal(t, make(map[string]string), actual)

	// Verify sibling namespace is not allowed
	actual = make(map[string]string)
	err = nlm.ParseJson([]byte(`{".node.kubernetes.io/key3": "value3"}`), actual)
	assert.Error(t, err)
	assert.Equal(t, make(map[string]string), actual)

	// Verify sibling namespace is not allowed
	actual = make(map[string]string)
	err = nlm.ParseJson([]byte(`{"ode.kubernetes.io/key3": "value3"}`), actual)
	assert.Error(t, err)
	assert.Equal(t, make(map[string]string), actual)

	// Verify changing namespace suffix is not allowed
	actual = make(map[string]string)
	err = nlm.ParseJson([]byte(`{"node.kubernetes.io./key3": "value3"}`), actual)
	assert.Error(t, err)
	assert.Equal(t, make(map[string]string), actual)

	// Verify changing namespace suffix is not allowed
	actual = make(map[string]string)
	err = nlm.ParseJson([]byte(`{"node.kubernetes.io.a/key3": "value3"}`), actual)
	assert.Error(t, err)
	assert.Equal(t, make(map[string]string), actual)

	// Verify label is required
	actual = make(map[string]string)
	err = nlm.ParseJson([]byte(`{"node.kubernetes.io": "value3"}`), actual)
	assert.Error(t, err)
	assert.Equal(t, make(map[string]string), actual)

	// Verify label is required
	actual = make(map[string]string)
	err = nlm.ParseJson([]byte(`{"node.kubernetes.io/": "value3"}`), actual)
	assert.Error(t, err)
	assert.Equal(t, make(map[string]string), actual)

	actual = make(map[string]string)
	err = nlm.ParseJson([]byte(`{"node.kubernetes.io /a": "value3"}`), actual)
	assert.Error(t, err)
	assert.Equal(t, make(map[string]string), actual)
}

func writeJsonFile(t *testing.T, tempDir string, name string, json string) {
	err := ioutil.WriteFile(path.Join(tempDir, name), []byte(json), 0664)
	assert.NoError(t, err)
}
