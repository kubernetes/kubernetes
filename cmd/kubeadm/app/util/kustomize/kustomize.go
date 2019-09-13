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

	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	yamlutil "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/cli-runtime/pkg/kustomize"
	"sigs.k8s.io/kustomize/pkg/constants"
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/loader"
	"sigs.k8s.io/kustomize/pkg/patch"
	"sigs.k8s.io/kustomize/pkg/types"
	"sigs.k8s.io/yaml"
)

// Manager define a manager that allow access to kustomize capabilities
type Manager struct {
	kustomizeDir          string
	kustomizationFile     *types.Kustomization
	strategicMergePatches strategicMergeSlice
	json6902Patches       json6902Slice
}

var (
	lock      = &sync.Mutex{}
	instances = map[string]*Manager{}
)

// GetManager return the KustomizeManager singleton instance
// NB. this is done at singleton instance level because kubeadm has a unique pool
// of patches that are applied to different content, at different time
func GetManager(kustomizeDir string) (*Manager, error) {
	lock.Lock()
	defer lock.Unlock()

	// if the instance does not exists, create it
	if _, ok := instances[kustomizeDir]; !ok {
		km := &Manager{
			kustomizeDir: kustomizeDir,
		}

		// Create a loader that mimics the behavior of kubectl kustomize, including support for reading from
		// a local folder or git repository like git@github.com:someOrg/someRepo.git or https://github.com/someOrg/someRepo?ref=someHash
		// in order to do so you must use ldr.Root() instead of km.kustomizeDir and ldr.Load instead of other ways to read files
		fSys := fs.MakeRealFS()
		ldr, err := loader.NewLoader(km.kustomizeDir, fSys)
		if err != nil {
			return nil, err
		}
		defer ldr.Cleanup()

		// read the Kustomization file and all the patches it is
		// referencing (either stategicMerge or json6902 patches)
		if err := km.loadFromKustomizationFile(ldr); err != nil {
			return nil, err
		}

		// if a Kustomization file was not found, kubeadm creates
		// one using all the patches in the folder; however in this
		// case only stategicMerge patches are supported
		if km.kustomizationFile == nil {
			km.kustomizationFile = &types.Kustomization{}
			if err := km.loadFromFolder(ldr); err != nil {
				return nil, err
			}
		}

		instances[kustomizeDir] = km
	}

	return instances[kustomizeDir], nil
}

// loadFromKustomizationFile reads a Kustomization file and all the patches it is
// referencing (either stategicMerge or json6902 patches)
func (km *Manager) loadFromKustomizationFile(ldr ifc.Loader) error {
	// Kustomize support different KustomizationFileNames, so we try to read all
	var content []byte
	match := 0
	for _, kf := range constants.KustomizationFileNames {
		c, err := ldr.Load(kf)
		if err == nil {
			match++
			content = c
		}
	}

	// if no kustomization file is found return
	if match == 0 {
		return nil
	}

	// if more that one kustomization file is found, return error
	if match > 1 {
		return errors.Errorf("Found multiple kustomization files under: %s\n", ldr.Root())
	}

	// Decode the kustomization file
	decoder := yamlutil.NewYAMLOrJSONDecoder(bytes.NewReader(content), 1024)
	var k = &types.Kustomization{}
	if err := decoder.Decode(k); err != nil {
		return errors.Wrap(err, "Error decoding kustomization file")
	}
	km.kustomizationFile = k

	// gets all the strategic merge patches
	for _, f := range km.kustomizationFile.PatchesStrategicMerge {
		smp, err := newStrategicMergeSliceFromFile(ldr, string(f))
		if err != nil {
			return err
		}
		km.strategicMergePatches = append(km.strategicMergePatches, smp...)
	}

	// gets all the json6902 patches
	for _, f := range km.kustomizationFile.PatchesJson6902 {
		jp, err := newJSON6902FromFile(f, ldr, f.Path)
		if err != nil {
			return err
		}
		km.json6902Patches = append(km.json6902Patches, jp)
	}

	return nil
}

// loadFromFolder returns all the stategicMerge patches in a folder
func (km *Manager) loadFromFolder(ldr ifc.Loader) error {
	files, err := ioutil.ReadDir(ldr.Root())
	if err != nil {
		return err
	}
	for _, fileInfo := range files {
		if fileInfo.IsDir() {
			continue
		}

		smp, err := newStrategicMergeSliceFromFile(ldr, fileInfo.Name())
		if err != nil {
			return err
		}
		km.strategicMergePatches = append(km.strategicMergePatches, smp...)
	}
	return nil
}

// Kustomize apply a set of patches to a resource.
// Portions of the kustomize logic in this function are taken from the kubernetes-sigs/kind project
func (km *Manager) Kustomize(data []byte) ([]byte, error) {
	// parse the resource to kustomize
	decoder := yamlutil.NewYAMLOrJSONDecoder(bytes.NewReader(data), 1024)
	var resource *unstructured.Unstructured
	if err := decoder.Decode(&resource); err != nil {
		return nil, err
	}

	// get patches corresponding to this resource
	strategicMerge := km.strategicMergePatches.filterByResource(resource)
	json6902 := km.json6902Patches.filterByResource(resource)

	// if there are no patches, for the target resources, exit
	patchesCnt := len(strategicMerge) + len(json6902)
	if patchesCnt == 0 {
		return data, nil
	}

	fmt.Printf("[kustomize] Applying %d patches to %s Resource=%s/%s\n", patchesCnt, resource.GroupVersionKind(), resource.GetNamespace(), resource.GetName())

	// create an in memory fs to use for the kustomization
	memFS := fs.MakeFakeFS()

	fakeDir := "/"
	// for Windows we need this to be a drive because kustomize uses filepath.Abs()
	// which will add a drive letter if there is none. which drive letter is
	// unimportant as the path is on the fake filesystem anyhow
	if runtime.GOOS == "windows" {
		fakeDir = `C:\`
	}

	// writes the resource to a file in the temp file system
	b, err := yaml.Marshal(resource)
	if err != nil {
		return nil, err
	}
	name := "resource.yaml"
	memFS.WriteFile(filepath.Join(fakeDir, name), b)

	km.kustomizationFile.Resources = []string{name}

	// writes strategic merge patches to files in the temp file system
	km.kustomizationFile.PatchesStrategicMerge = []patch.StrategicMerge{}
	for i, p := range strategicMerge {
		b, err := yaml.Marshal(p)
		if err != nil {
			return nil, err
		}
		name := fmt.Sprintf("patch-%d.yaml", i)
		memFS.WriteFile(filepath.Join(fakeDir, name), b)

		km.kustomizationFile.PatchesStrategicMerge = append(km.kustomizationFile.PatchesStrategicMerge, patch.StrategicMerge(name))
	}

	// writes json6902 patches to files in the temp file system
	km.kustomizationFile.PatchesJson6902 = []patch.Json6902{}
	for i, p := range json6902 {
		name := fmt.Sprintf("patchjson-%d.yaml", i)
		memFS.WriteFile(filepath.Join(fakeDir, name), []byte(p.Patch))

		km.kustomizationFile.PatchesJson6902 = append(km.kustomizationFile.PatchesJson6902, patch.Json6902{Target: p.Target, Path: name})
	}

	// writes the kustomization file to the temp file system
	kbytes, err := yaml.Marshal(km.kustomizationFile)
	if err != nil {
		return nil, err
	}
	memFS.WriteFile(filepath.Join(fakeDir, "kustomization.yaml"), kbytes)

	// Finally customize the target resource
	var out bytes.Buffer
	if err := kustomize.RunKustomizeBuild(&out, memFS, fakeDir); err != nil {
		return nil, err
	}

	return out.Bytes(), nil
}
