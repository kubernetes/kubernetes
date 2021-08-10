// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package utils

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"time"

	"sigs.k8s.io/kustomize/api/konfig"
	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/filesys"
	"sigs.k8s.io/yaml"
)

const (
	idAnnotation       = "kustomize.config.k8s.io/id"
	HashAnnotation     = "kustomize.config.k8s.io/needs-hash"
	BehaviorAnnotation = "kustomize.config.k8s.io/behavior"
)

func GoBin() string {
	return filepath.Join(runtime.GOROOT(), "bin", "go")
}

// DeterminePluginSrcRoot guesses where the user
// has her ${g}/${v}/$lower(${k})/${k}.go files.
func DeterminePluginSrcRoot(fSys filesys.FileSystem) (string, error) {
	return konfig.FirstDirThatExistsElseError(
		"plugin src root", fSys, []konfig.NotedFunc{
			{
				Note: "relative to unit test",
				F: func() string {
					return filepath.Clean(
						filepath.Join(
							os.Getenv("PWD"),
							"..", "..",
							konfig.RelPluginHome))
				},
			},
			{
				Note: "relative to unit test (internal pkg)",
				F: func() string {
					return filepath.Clean(
						filepath.Join(
							os.Getenv("PWD"),
							"..", "..", "..", "..",
							konfig.RelPluginHome))
				},
			},
			{
				Note: "relative to api package",
				F: func() string {
					return filepath.Clean(
						filepath.Join(
							os.Getenv("PWD"),
							"..", "..", "..",
							konfig.RelPluginHome))
				},
			},
			{
				Note: "old style $GOPATH",
				F: func() string {
					return filepath.Join(
						os.Getenv("GOPATH"),
						"src", konfig.DomainName,
						konfig.ProgramName, konfig.RelPluginHome)
				},
			},
			{
				Note: "HOME with literal 'gopath'",
				F: func() string {
					return filepath.Join(
						konfig.HomeDir(), "gopath",
						"src", konfig.DomainName,
						konfig.ProgramName, konfig.RelPluginHome)
				},
			},
			{
				Note: "home directory",
				F: func() string {
					return filepath.Join(
						konfig.HomeDir(), konfig.DomainName,
						konfig.ProgramName, konfig.RelPluginHome)
				},
			},
		})
}

// FileYoungerThan returns true if the file both exists and has an
// age is <= the Duration argument.
func FileYoungerThan(path string, d time.Duration) bool {
	fi, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return false
		}
	}
	return time.Since(fi.ModTime()) <= d
}

// FileModifiedAfter returns true if the file both exists and was
// modified after the given time..
func FileModifiedAfter(path string, t time.Time) bool {
	fi, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return false
		}
	}
	return fi.ModTime().After(t)
}

func FileExists(path string) bool {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return false
		}
	}
	return true
}

// GetResMapWithIDAnnotation returns a new copy of the given ResMap with the ResIds annotated in each Resource
func GetResMapWithIDAnnotation(rm resmap.ResMap) (resmap.ResMap, error) {
	inputRM := rm.DeepCopy()
	for _, r := range inputRM.Resources() {
		idString, err := yaml.Marshal(r.CurId())
		if err != nil {
			return nil, err
		}
		annotations := r.GetAnnotations()
		annotations[idAnnotation] = string(idString)
		if err = r.SetAnnotations(annotations); err != nil {
			return nil, err
		}
	}
	return inputRM, nil
}

// UpdateResMapValues updates the Resource value in the given ResMap
// with the emitted Resource values in output.
func UpdateResMapValues(pluginName string, h *resmap.PluginHelpers, output []byte, rm resmap.ResMap) error {
	mapFactory := h.ResmapFactory()
	resFactory := mapFactory.RF()
	resources, err := resFactory.SliceFromBytes(output)
	if err != nil {
		return err
	}
	// Don't use resources here, or error message will be unfriendly to plugin builders
	newMap, err := mapFactory.NewResMapFromBytes([]byte{})
	if err != nil {
		return err
	}

	for _, r := range resources {
		// stale--not manipulated by plugin transformers
		if err = removeIDAnnotation(r); err != nil {
			return err
		}

		// Add to the new map, checking for duplicates
		if err := newMap.Append(r); err != nil {
			prettyID, err := json.Marshal(r.CurId())
			if err != nil {
				prettyID = []byte(r.CurId().String())
			}
			return fmt.Errorf("plugin %s generated duplicate resource: %s", pluginName, prettyID)
		}

		// Add to or update the old map
		oldIdx, err := rm.GetIndexOfCurrentId(r.CurId())
		if err != nil {
			return err
		}
		if oldIdx != -1 {
			rm.GetByIndex(oldIdx).ResetRNode(r)
		} else {
			if err := rm.Append(r); err != nil {
				return err
			}
		}
	}

	// Remove items the transformer deleted from the old map
	for _, id := range rm.AllIds() {
		newIdx, _ := newMap.GetIndexOfCurrentId(id)
		if newIdx == -1 {
			if err = rm.Remove(id); err != nil {
				return err
			}
		}
	}

	return nil
}

func removeIDAnnotation(r *resource.Resource) error {
	// remove the annotation set by Kustomize to track the resource
	annotations := r.GetAnnotations()
	delete(annotations, idAnnotation)
	return r.SetAnnotations(annotations)
}

// UpdateResourceOptions updates the generator options for each resource in the
// given ResMap based on plugin provided annotations.
func UpdateResourceOptions(rm resmap.ResMap) (resmap.ResMap, error) {
	for _, r := range rm.Resources() {
		// Disable name hashing by default and require plugin to explicitly
		// request it for each resource.
		annotations := r.GetAnnotations()
		behavior := annotations[BehaviorAnnotation]
		var needsHash bool
		if val, ok := annotations[HashAnnotation]; ok {
			b, err := strconv.ParseBool(val)
			if err != nil {
				return nil, fmt.Errorf(
					"the annotation %q contains an invalid value (%q)",
					HashAnnotation, val)
			}
			needsHash = b
		}
		delete(annotations, HashAnnotation)
		delete(annotations, BehaviorAnnotation)
		if err := r.SetAnnotations(annotations); err != nil {
			return nil, err
		}
		r.SetOptions(types.NewGenArgs(
			&types.GeneratorArgs{
				Behavior: behavior,
				Options:  &types.GeneratorOptions{DisableNameSuffixHash: !needsHash}}))
	}
	return rm, nil
}
