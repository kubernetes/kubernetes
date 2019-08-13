/*
Copyright 2019 The Kubernetes Authors.

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

// Package kustomize contains helpers for working with embedded kustomize commands
package kustomize

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"runtime"
	"sync"

	"k8s.io/cli-runtime/pkg/kustomize"
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/loader"
)

// Manager define a manager that allow access to kustomize capabilities
type Manager struct {
	kustomizeDir string
	us           UnstructuredSlice
}

var (
	lock      = &sync.Mutex{}
	instances = map[string]*Manager{}
)

// GetManager return the KustomizeManager singleton instance
func GetManager(kustomizeDir string) (*Manager, error) {
	lock.Lock()
	defer lock.Unlock()

	// if the instance does not exists, create it
	if _, ok := instances[kustomizeDir]; !ok {
		km := &Manager{
			kustomizeDir: kustomizeDir,
		}

		// loads the UnstructuredSlice with all the patches into the Manager
		// NB. this is done at singleton instance level because kubeadm has a unique pool
		// of patches that are applied to different content, at different time
		if err := km.getUnstructuredSlice(); err != nil {
			return nil, err
		}

		instances[kustomizeDir] = km
	}

	return instances[kustomizeDir], nil
}

// getUnstructuredSlice returns a UnstructuredSlice with all the patches.
func (km *Manager) getUnstructuredSlice() error {
	// kubeadm does not require a kustomization.yaml file listing all the resources/patches, so it is necessary
	// to rebuild the list of patches manually
	// TODO: make this git friendly - currently this works only for patches in local folders -
	files, err := ioutil.ReadDir(km.kustomizeDir)
	if err != nil {
		return err
	}

	var paths = []string{}
	for _, file := range files {
		if file.IsDir() {
			continue
		}
		paths = append(paths, file.Name())
	}

	// Create a loader that mimics the behavior of kubectl kustomize, including support for reading from
	// a local git repository like git@github.com:someOrg/someRepo.git or https://github.com/someOrg/someRepo?ref=someHash
	fSys := fs.MakeRealFS()
	ldr, err := loader.NewLoader(km.kustomizeDir, fSys)
	if err != nil {
		return err
	}
	defer ldr.Cleanup()

	// read all the kustomizations and build the UnstructuredSlice
	us, err := NewUnstructuredSliceFromFiles(ldr, paths)
	if err != nil {
		return err
	}

	km.us = us
	return nil
}

// Kustomize apply a set of patches to a resource.
// Portions of the kustomize logic in this function are taken from the kubernetes-sigs/kind project
func (km *Manager) Kustomize(res []byte) ([]byte, error) {
	// create a loader that mimics the behavior of kubectl kustomize
	// and converts the resource into a UnstructuredSlice
	// Nb. in kubeadm we are controlling resource generation, and so we
	// we are expecting 1 object into each resource, eg. the static pod.
	// Nevertheless, this code is ready for more than one object per resource
	resList, err := NewUnstructuredSliceFromBytes(res)
	if err != nil {
		return nil, err
	}

	// create a list of resource and corresponding patches
	var resources, patches UnstructuredSlice
	for _, r := range resList {
		resources = append(resources, r)

		resourcePatches := km.us.FilterResource(r.GroupVersionKind(), r.GetNamespace(), r.GetName())
		patches = append(patches, resourcePatches...)
	}

	fmt.Printf("[kustomize] Applying %d patches\n", len(patches))

	// if there are no patches, for the target resources, exit
	if len(patches) == 0 {
		return res, nil
	}

	// create an in memory fs to use for the kustomization
	memFS := fs.MakeFakeFS()

	var kustomization bytes.Buffer
	fakeDir := "/"
	// for Windows we need this to be a drive because kustomize uses filepath.Abs()
	// which will add a drive letter if there is none. which drive letter is
	// unimportant as the path is on the fake filesystem anyhow
	if runtime.GOOS == "windows" {
		fakeDir = `C:\`
	}

	// write resources and patches to the in memory fs, generate the kustomization.yaml
	// that ties everything together
	kustomization.WriteString("resources:\n")
	for i, r := range resources {
		b, err := r.MarshalJSON()
		if err != nil {
			return nil, err
		}

		name := fmt.Sprintf("resource-%d.json", i)
		_ = memFS.WriteFile(filepath.Join(fakeDir, name), b)
		fmt.Fprintf(&kustomization, " - %s\n", name)
	}

	kustomization.WriteString("patches:\n")
	for i, p := range patches {
		b, err := p.MarshalJSON()
		if err != nil {
			return nil, err
		}

		name := fmt.Sprintf("patch-%d.json", i)
		_ = memFS.WriteFile(filepath.Join(fakeDir, name), b)
		fmt.Fprintf(&kustomization, " - %s\n", name)
	}

	memFS.WriteFile(filepath.Join(fakeDir, "kustomization.yaml"), kustomization.Bytes())

	// Finally customize the target resource
	var out bytes.Buffer
	if err := kustomize.RunKustomizeBuild(&out, memFS, fakeDir); err != nil {
		return nil, err
	}

	return out.Bytes(), nil
}
