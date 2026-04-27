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

package subpath

import "os"

// Interface defines the set of methods all subpathers must implement
type Interface interface {
	// CleanSubPaths removes any bind-mounts created by PrepareSafeSubpath in given
	// pod volume directory.
	CleanSubPaths(poodDir string, volumeName string) error

	// PrepareSafeSubpath does everything that's necessary to prepare a subPath
	// that's 1) inside given volumePath and 2) immutable after this call.
	//
	// newHostPath - location of prepared subPath. It should be used instead of
	// hostName when running the container.
	// cleanupAction - action to run when the container is running or it failed to start.
	//
	// CleanupAction must be called immediately after the container with given
	// subpath starts. On the other hand, Interface.CleanSubPaths must be called
	// when the pod finishes.
	PrepareSafeSubpath(subPath Subpath) (newHostPath string, cleanupAction func(), err error)

	// SafeMakeDir creates subdir within given base. It makes sure that the
	// created directory does not escape given base directory mis-using
	// symlinks. Note that the function makes sure that it creates the directory
	// somewhere under the base, nothing else. E.g. if the directory already
	// exists, it may exist outside of the base due to symlinks.
	// This method should be used if the directory to create is inside volume
	// that's under user control. User must not be able to use symlinks to
	// escape the volume to create directories somewhere else.
	SafeMakeDir(subdir string, base string, perm os.FileMode) error
}

// Subpath defines the attributes of a subpath
type Subpath struct {
	// index of the VolumeMount for this container
	VolumeMountIndex int

	// Full path to the subpath directory on the host
	Path string

	// name of the volume that is a valid directory name.
	VolumeName string

	// Full path to the volume path
	VolumePath string

	// Path to the pod's directory, including pod UID
	PodDir string

	// Name of the container
	ContainerName string
}

// Compile time-check for all implementers of subpath interface
var _ Interface = &subpath{}
var _ Interface = &FakeSubpath{}

// FakeSubpath is a subpather implementation for testing
type FakeSubpath struct{}

// PrepareSafeSubpath is a fake implementation of PrepareSafeSubpath. Always returns
// newHostPath == subPath.Path
func (fs *FakeSubpath) PrepareSafeSubpath(subPath Subpath) (newHostPath string, cleanupAction func(), err error) {
	return subPath.Path, nil, nil
}

// CleanSubPaths is a fake implementation of CleanSubPaths. It is a noop
func (fs *FakeSubpath) CleanSubPaths(podDir string, volumeName string) error {
	return nil
}

// SafeMakeDir is a fake implementation of SafeMakeDir. It is a noop
func (fs *FakeSubpath) SafeMakeDir(pathname string, base string, perm os.FileMode) error {
	return nil
}
