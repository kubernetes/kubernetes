/*
Copyright 2018 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	v1 "k8s.io/api/core/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

// getNestedMountpoints returns a list of mountpoint directories that should be created
// for the volume indicated by name.
// note: the returned list is relative to baseDir
func getNestedMountpoints(name, baseDir string, pod v1.Pod) ([]string, error) {
	var retval []string
	checkContainer := func(container *v1.Container) error {
		var allMountPoints []string // all mount points in this container
		var myMountPoints []string  // mount points that match name
		for _, vol := range container.VolumeMounts {
			cleaned := filepath.Clean(vol.MountPath)
			allMountPoints = append(allMountPoints, cleaned)
			if vol.Name == name {
				myMountPoints = append(myMountPoints, cleaned)
			}
		}
		sort.Strings(allMountPoints)
		parentPrefix := ".." + string(os.PathSeparator)
		// Examine each place where this volume is mounted
		for _, myMountPoint := range myMountPoints {
			if strings.HasPrefix(myMountPoint, parentPrefix) {
				// Don't let a container trick us into creating directories outside of its rootfs
				return fmt.Errorf("invalid container mount point %v", myMountPoint)
			}
			myMPSlash := myMountPoint + string(os.PathSeparator)
			// The previously found nested mountpoints.
			// NOTE: We can't simply rely on sort.Strings to have all the mountpoints sorted and
			// grouped. For example, the following strings are sorted in this exact order:
			// /dir/nested, /dir/nested-vol, /dir/nested.vol, /dir/nested/double, /dir/nested2
			// The issue is a bit worse for Windows paths, since the \'s value is higher than /'s:
			// \dir\nested, \dir\nested-vol, \dir\nested.vol, \dir\nested2, \dir\nested\double
			// Because of this, we should use a list of previously mounted mountpoints, rather than only one.
			prevNestedMPs := []string{}
			// examine each mount point to see if it's nested beneath this volume
			// (but skip any that are double-nested beneath this volume)
			// For example, if this volume is mounted as /dir and other volumes are mounted
			//              as /dir/nested and /dir/nested/other, only create /dir/nested.
			for _, mp := range allMountPoints {
				if !strings.HasPrefix(mp, myMPSlash) {
					continue // skip -- not nested beneath myMountPoint
				}

				isNested := false
				for _, prevNestedMP := range prevNestedMPs {
					if strings.HasPrefix(mp, prevNestedMP) {
						isNested = true
						break
					}
				}
				if isNested {
					continue // skip -- double nested beneath myMountPoint
				}
				// since this mount point is nested, remember it so that we can check that following ones aren't nested beneath this one
				prevNestedMPs = append(prevNestedMPs, mp+string(os.PathSeparator))
				retval = append(retval, mp[len(myMPSlash):])
			}
		}
		return nil
	}

	var retErr error
	podutil.VisitContainers(&pod.Spec, podutil.AllFeatureEnabledContainers(), func(c *v1.Container, containerType podutil.ContainerType) bool {
		retErr = checkContainer(c)
		return retErr == nil
	})
	if retErr != nil {
		return nil, retErr
	}

	return retval, nil
}

// MakeNestedMountpoints creates mount points in baseDir for volumes mounted beneath name
func MakeNestedMountpoints(name, baseDir string, pod v1.Pod) error {
	dirs, err := getNestedMountpoints(name, baseDir, pod)
	if err != nil {
		return err
	}
	for _, dir := range dirs {
		err := os.MkdirAll(filepath.Join(baseDir, dir), 0755)
		if err != nil {
			return fmt.Errorf("unable to create nested volume mountpoints: %v", err)
		}
	}
	return nil
}
