/*
Copyright 2015 The Kubernetes Authors.

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

package io

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/runtime"
)

// ioTimeout is the timeout to use for io operations
const ioTimeout time.Duration = 5

// Lstat wraps os.Lstat in a timeout.
func Lstat(path string) (os.FileInfo, error) {
	return internalLstatTimeout(path, ioTimeout)
}

func internalLstatTimeout(path string, timeout time.Duration) (os.FileInfo, error) {
	var (
		timer      = time.NewTimer(ioTimeout)
		resChannel = make(chan os.FileInfo)
		errChannel = make(chan error)
	)

	defer timer.Stop()

	go func() {
		fileInfo, err := os.Lstat(path)
		if err != nil {
			errChannel <- err
		} else {
			resChannel <- fileInfo
		}
	}()

	select {
	case <-timer.C:
		return nil, fmt.Errorf("Lstat on %v timed out after %v", path, timeout)
	case err := <-errChannel:
		return nil, err
	case res := <-resChannel:
		return res, nil
	}
}

// ReadDir is similar to ioutil.ReadDir, but returns an error after a timeout.
func ReadDir(path string) ([]os.FileInfo, error) {
	return internalReadDirTimeout(path, ioTimeout)
}

func internalReadDirTimeout(path string, timeout time.Duration) ([]os.FileInfo, error) {
	timer := time.NewTimer(timeout)
	resChannel := make(chan []os.FileInfo)
	errChannel := make(chan error)

	defer timer.Stop()

	go func() {
		entries, err := ioutil.ReadDir(path)
		if err != nil {
			errChannel <- err
		} else {
			resChannel <- entries
		}
	}()

	select {
	case <-timer.C:
		return nil, fmt.Errorf("ReadDir on %v timed out after %v", path, timeout)
	case err := <-errChannel:
		return nil, err
	case res := <-resChannel:
		return res, nil
	}
}

// ReadDirNoExit reads the directory named by dirname and returns:
// 1.  A sparse list of directory entries that were able to be Lstat'd
// 2.  A sparse list of errors encountered when Lstat-ing entries
// 3.  An overall error if the directory couldn't be opened
//
// ReadDirNoExit uses an Lstat call that implements a timeout; if Lstat
// on a directory entry times out, an error will be present in the error
// list.
func ReadDirNoExit(dirname string) ([]os.FileInfo, []error, error) {
	f, err := os.Open(dirname)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	names, err := f.Readdirnames(-1)
	list := make([]os.FileInfo, 0, len(names))
	errs := make([]error, 0, len(names))
	for _, filename := range names {
		fip, statErr := Lstat(filepath.Join(dirname, filename))
		if os.IsNotExist(statErr) {
			// File disappeared between readdir + stat.
			// Just treat it as if it didn't exist.
			continue
		}

		list = append(list, fip)
		errs = append(errs, statErr)
	}

	return list, errs, nil
}

// LoadPodFromFile will read, decode, and return a Pod from a file.
func LoadPodFromFile(filePath string) (*api.Pod, error) {
	if filePath == "" {
		return nil, fmt.Errorf("file path not specified")
	}
	podDef, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file path %s: %+v", filePath, err)
	}
	if len(podDef) == 0 {
		return nil, fmt.Errorf("file was empty: %s", filePath)
	}
	pod := &api.Pod{}

	codec := api.Codecs.LegacyCodec(registered.GroupOrDie(api.GroupName).GroupVersion)
	if err := runtime.DecodeInto(codec, podDef, pod); err != nil {
		return nil, fmt.Errorf("failed decoding file: %v", err)
	}
	return pod, nil
}

// SavePodToFile will encode and save a pod to a given path & permissions
func SavePodToFile(pod *api.Pod, filePath string, perm os.FileMode) error {
	if filePath == "" {
		return fmt.Errorf("file path not specified")
	}
	codec := api.Codecs.LegacyCodec(registered.GroupOrDie(api.GroupName).GroupVersion)
	data, err := runtime.Encode(codec, pod)
	if err != nil {
		return fmt.Errorf("failed encoding pod: %v", err)
	}
	return ioutil.WriteFile(filePath, data, perm)
}
