/*
   Copyright The containerd Authors.

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

package cgroup1

import (
	"errors"
	"fmt"
	"path/filepath"
)

type Path func(subsystem Name) (string, error)

func RootPath(subsystem Name) (string, error) {
	return "/", nil
}

// StaticPath returns a static path to use for all cgroups
func StaticPath(path string) Path {
	return func(_ Name) (string, error) {
		return path, nil
	}
}

// NestedPath will nest the cgroups based on the calling processes cgroup
// placing its child processes inside its own path
func NestedPath(suffix string) Path {
	paths, err := ParseCgroupFile("/proc/self/cgroup")
	if err != nil {
		return errorPath(err)
	}
	return existingPath(paths, suffix)
}

// PidPath will return the correct cgroup paths for an existing process running inside a cgroup
// This is commonly used for the Load function to restore an existing container
func PidPath(pid int) Path {
	p := fmt.Sprintf("/proc/%d/cgroup", pid)
	paths, err := ParseCgroupFile(p)
	if err != nil {
		return errorPath(fmt.Errorf("parse cgroup file %s: %w", p, err))
	}
	return existingPath(paths, "")
}

// ErrControllerNotActive is returned when a controller is not supported or enabled
var ErrControllerNotActive = errors.New("controller is not supported")

func existingPath(paths map[string]string, suffix string) Path {
	// localize the paths based on the root mount dest for nested cgroups
	for n, p := range paths {
		dest, err := getCgroupDestination(n)
		if err != nil {
			return errorPath(err)
		}
		rel, err := filepath.Rel(dest, p)
		if err != nil {
			return errorPath(err)
		}
		if rel == "." {
			rel = dest
		}
		paths[n] = filepath.Join("/", rel)
	}
	return func(name Name) (string, error) {
		root, ok := paths[string(name)]
		if !ok {
			if root, ok = paths["name="+string(name)]; !ok {
				return "", ErrControllerNotActive
			}
		}
		if suffix != "" {
			return filepath.Join(root, suffix), nil
		}
		return root, nil
	}
}

func subPath(path Path, subName string) Path {
	return func(name Name) (string, error) {
		p, err := path(name)
		if err != nil {
			return "", err
		}
		return filepath.Join(p, subName), nil
	}
}

func errorPath(err error) Path {
	return func(_ Name) (string, error) {
		return "", err
	}
}
