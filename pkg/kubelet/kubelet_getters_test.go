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
)

func TestKubeletDirs(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	root := kubelet.rootDirectory

	var exp, got string

	got = kubelet.getPodsDir()
	exp = filepath.Join(root, "pods")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPluginsDir()
	exp = filepath.Join(root, "plugins")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPluginDir("foobar")
	exp = filepath.Join(root, "plugins/foobar")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodDir("abc123")
	exp = filepath.Join(root, "pods/abc123")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodVolumesDir("abc123")
	exp = filepath.Join(root, "pods/abc123/volumes")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodVolumeDir("abc123", "plugin", "foobar")
	exp = filepath.Join(root, "pods/abc123/volumes/plugin/foobar")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodPluginsDir("abc123")
	exp = filepath.Join(root, "pods/abc123/plugins")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodPluginDir("abc123", "foobar")
	exp = filepath.Join(root, "pods/abc123/plugins/foobar")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodContainerDir("abc123", "def456")
	exp = filepath.Join(root, "pods/abc123/containers/def456")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}
}

func TestKubeletDirsCompat(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	root := kubelet.rootDirectory
	if err := os.MkdirAll(root, 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}

	var exp, got string

	// Old-style pod dir.
	if err := os.MkdirAll(fmt.Sprintf("%s/oldpod", root), 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}
	// New-style pod dir.
	if err := os.MkdirAll(fmt.Sprintf("%s/pods/newpod", root), 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}
	// Both-style pod dir.
	if err := os.MkdirAll(fmt.Sprintf("%s/bothpod", root), 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}
	if err := os.MkdirAll(fmt.Sprintf("%s/pods/bothpod", root), 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}

	got = kubelet.getPodDir("oldpod")
	exp = filepath.Join(root, "oldpod")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodDir("newpod")
	exp = filepath.Join(root, "pods/newpod")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodDir("bothpod")
	exp = filepath.Join(root, "pods/bothpod")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodDir("neitherpod")
	exp = filepath.Join(root, "pods/neitherpod")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	root = kubelet.getPodDir("newpod")

	// Old-style container dir.
	if err := os.MkdirAll(fmt.Sprintf("%s/oldctr", root), 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}
	// New-style container dir.
	if err := os.MkdirAll(fmt.Sprintf("%s/containers/newctr", root), 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}
	// Both-style container dir.
	if err := os.MkdirAll(fmt.Sprintf("%s/bothctr", root), 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}
	if err := os.MkdirAll(fmt.Sprintf("%s/containers/bothctr", root), 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}

	got = kubelet.getPodContainerDir("newpod", "oldctr")
	exp = filepath.Join(root, "oldctr")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodContainerDir("newpod", "newctr")
	exp = filepath.Join(root, "containers/newctr")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodContainerDir("newpod", "bothctr")
	exp = filepath.Join(root, "containers/bothctr")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodContainerDir("newpod", "neitherctr")
	exp = filepath.Join(root, "containers/neitherctr")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}
}
