/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestKubeletDirs(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	root := kubelet.rootDirectory

	var exp, got string

	got = kubelet.getPodsDir()
	exp = filepath.Join(root, "pods")
	assert.Equal(t, exp, got)

	got = kubelet.getPluginsDir()
	exp = filepath.Join(root, "plugins")
	assert.Equal(t, exp, got)

	got = kubelet.getPluginDir("foobar")
	exp = filepath.Join(root, "plugins/foobar")
	assert.Equal(t, exp, got)

	got = kubelet.getPodDir("abc123")
	exp = filepath.Join(root, "pods/abc123")
	assert.Equal(t, exp, got)

	got = kubelet.getPodVolumesDir("abc123")
	exp = filepath.Join(root, "pods/abc123/volumes")
	assert.Equal(t, exp, got)

	got = kubelet.getPodVolumeDir("abc123", "plugin", "foobar")
	exp = filepath.Join(root, "pods/abc123/volumes/plugin/foobar")
	assert.Equal(t, exp, got)

	got = kubelet.getPodPluginsDir("abc123")
	exp = filepath.Join(root, "pods/abc123/plugins")
	assert.Equal(t, exp, got)

	got = kubelet.getPodPluginDir("abc123", "foobar")
	exp = filepath.Join(root, "pods/abc123/plugins/foobar")
	assert.Equal(t, exp, got)

	got = kubelet.getPodContainerDir("abc123", "def456")
	exp = filepath.Join(root, "pods/abc123/containers/def456")
	assert.Equal(t, exp, got)
}

func TestKubeletDirsCompat(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	root := kubelet.rootDirectory
	require.NoError(t, os.MkdirAll(root, 0750), "can't mkdir(%q)", root)

	// Old-style pod dir.
	require.NoError(t, os.MkdirAll(fmt.Sprintf("%s/oldpod", root), 0750), "can't mkdir(%q)", root)

	// New-style pod dir.
	require.NoError(t, os.MkdirAll(fmt.Sprintf("%s/pods/newpod", root), 0750), "can't mkdir(%q)", root)

	// Both-style pod dir.
	require.NoError(t, os.MkdirAll(fmt.Sprintf("%s/bothpod", root), 0750), "can't mkdir(%q)", root)
	require.NoError(t, os.MkdirAll(fmt.Sprintf("%s/pods/bothpod", root), 0750), "can't mkdir(%q)", root)

	assert.Equal(t, filepath.Join(root, "oldpod"), kubelet.getPodDir("oldpod"))
	assert.Equal(t, filepath.Join(root, "pods/newpod"), kubelet.getPodDir("newpod"))
	assert.Equal(t, filepath.Join(root, "pods/bothpod"), kubelet.getPodDir("bothpod"))
	assert.Equal(t, filepath.Join(root, "pods/neitherpod"), kubelet.getPodDir("neitherpod"))

	root = kubelet.getPodDir("newpod")

	// Old-style container dir.
	require.NoError(t, os.MkdirAll(fmt.Sprintf("%s/oldctr", root), 0750), "can't mkdir(%q)", root)

	// New-style container dir.
	require.NoError(t, os.MkdirAll(fmt.Sprintf("%s/containers/newctr", root), 0750), "can't mkdir(%q)", root)

	// Both-style container dir.
	require.NoError(t, os.MkdirAll(fmt.Sprintf("%s/bothctr", root), 0750), "can't mkdir(%q)", root)
	require.NoError(t, os.MkdirAll(fmt.Sprintf("%s/containers/bothctr", root), 0750), "can't mkdir(%q)", root)

	assert.Equal(t, filepath.Join(root, "oldctr"), kubelet.getPodContainerDir("newpod", "oldctr"))
	assert.Equal(t, filepath.Join(root, "containers/newctr"), kubelet.getPodContainerDir("newpod", "newctr"))
	assert.Equal(t, filepath.Join(root, "containers/bothctr"), kubelet.getPodContainerDir("newpod", "bothctr"))
	assert.Equal(t, filepath.Join(root, "containers/neitherctr"), kubelet.getPodContainerDir("newpod", "neitherctr"))
}
