/*
Copyright 2014 Google Inc. All rights reserved.

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

// Reads the pod configuration from file or a directory of files.
package config

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/ghodss/yaml"
	"github.com/golang/glog"
)

type sourceFile struct {
	path    string
	updates chan<- interface{}
}

func NewSourceFile(path string, period time.Duration, updates chan<- interface{}) {
	config := &sourceFile{
		path:    path,
		updates: updates,
	}
	glog.V(1).Infof("Watching path %q", path)
	go util.Forever(config.run, period)
}

func (s *sourceFile) run() {
	if err := s.extractFromPath(); err != nil {
		glog.Errorf("Unable to read config path %q: %v", s.path, err)
	}
}

func (s *sourceFile) extractFromPath() error {
	path := s.path
	statInfo, err := os.Stat(path)
	if err != nil {
		if !os.IsNotExist(err) {
			return err
		}
		// Emit an update with an empty PodList to allow FileSource to be marked as seen
		s.updates <- kubelet.PodUpdate{[]api.Pod{}, kubelet.SET, kubelet.FileSource}
		return fmt.Errorf("path does not exist, ignoring")
	}

	switch {
	case statInfo.Mode().IsDir():
		pods, err := extractFromDir(path)
		if err != nil {
			return err
		}
		s.updates <- kubelet.PodUpdate{pods, kubelet.SET, kubelet.FileSource}

	case statInfo.Mode().IsRegular():
		pod, err := extractFromFile(path)
		if err != nil {
			return err
		}
		s.updates <- kubelet.PodUpdate{[]api.Pod{pod}, kubelet.SET, kubelet.FileSource}

	default:
		return fmt.Errorf("path is not a directory or file")
	}

	return nil
}

// Get as many pod configs as we can from a directory.  Return an error iff something
// prevented us from reading anything at all.  Do not return an error if only some files
// were problematic.
func extractFromDir(name string) ([]api.Pod, error) {
	dirents, err := filepath.Glob(filepath.Join(name, "[^.]*"))
	if err != nil {
		return nil, fmt.Errorf("glob failed: %v", err)
	}

	pods := make([]api.Pod, 0)
	if len(dirents) == 0 {
		return pods, nil
	}

	sort.Strings(dirents)
	for _, path := range dirents {
		statInfo, err := os.Stat(path)
		if err != nil {
			glog.V(1).Infof("Can't get metadata for %q: %v", path, err)
			continue
		}

		switch {
		case statInfo.Mode().IsDir():
			glog.V(1).Infof("Not recursing into config path %q", path)
		case statInfo.Mode().IsRegular():
			pod, err := extractFromFile(path)
			if err != nil {
				glog.V(1).Infof("Can't process config file %q: %v", path, err)
			} else {
				pods = append(pods, pod)
			}
		default:
			glog.V(1).Infof("Config path %q is not a directory or file: %v", path, statInfo.Mode())
		}
	}
	return pods, nil
}

func extractFromFile(filename string) (api.Pod, error) {
	var pod api.Pod

	glog.V(3).Infof("Reading config file %q", filename)
	file, err := os.Open(filename)
	if err != nil {
		return pod, err
	}
	defer file.Close()

	data, err := ioutil.ReadAll(file)
	if err != nil {
		return pod, err
	}

	// TODO: use api.Scheme.DecodeInto
	// This is awful.  DecodeInto() expects to find an APIObject, which
	// Manifest is not.  We keep reading manifest for now for compat, but
	// we will eventually change it to read Pod (at which point this all
	// becomes nicer).  Until then, we assert that the ContainerManifest
	// structure on disk is always v1beta1.  Read that, convert it to a
	// "current" ContainerManifest (should be ~identical), then convert
	// that to a Pod (which is a well-understood conversion).  This
	// avoids writing a v1beta1.ContainerManifest -> api.Pod
	// conversion which would be identical to the api.ContainerManifest ->
	// api.Pod conversion.
	oldManifest := &v1beta1.ContainerManifest{}
	if err := yaml.Unmarshal(data, oldManifest); err != nil {
		return pod, fmt.Errorf("can't unmarshal file %q: %v", filename, err)
	}
	newManifest := &api.ContainerManifest{}
	if err := api.Scheme.Convert(oldManifest, newManifest); err != nil {
		return pod, fmt.Errorf("can't convert pod from file %q: %v", filename, err)
	}
	if err := api.Scheme.Convert(newManifest, &pod); err != nil {
		return pod, fmt.Errorf("can't convert pod from file %q: %v", filename, err)
	}

	hostname, err := os.Hostname() //TODO: kubelet name would be better
	if err != nil {
		return pod, err
	}
	hostname = strings.ToLower(hostname)

	if len(pod.UID) == 0 {
		hasher := md5.New()
		fmt.Fprintf(hasher, "host:%s", hostname)
		fmt.Fprintf(hasher, "file:%s", filename)
		util.DeepHashObject(hasher, pod)
		pod.UID = types.UID(hex.EncodeToString(hasher.Sum(nil)[0:]))
		glog.V(5).Infof("Generated UID %q for pod %q from file %s", pod.UID, pod.Name, filename)
	}
	// This is required for backward compatibility, and should be removed once we
	// completely deprecate ContainerManifest.
	if len(pod.Name) == 0 {
		pod.Name = string(pod.UID)
	}
	if pod.Name, err = GeneratePodName(pod.Name); err != nil {
		return pod, err
	}
	glog.V(5).Infof("Generated Name %q for UID %q from file %s", pod.Name, pod.UID, filename)

	// Always overrides the namespace provided by the file.
	pod.Namespace = kubelet.NamespaceDefault
	glog.V(5).Infof("Using namespace %q for pod %q from file %s", pod.Namespace, pod.Name, filename)

	// TODO(dchen1107): BoundPod is not type of runtime.Object. Once we allow kubelet talks
	// about Pod directly, we can use SelfLinker defined in package: latest
	// Currently just simply follow the same format in resthandler.go
	pod.ObjectMeta.SelfLink = fmt.Sprintf("/api/v1beta2/pods/%s?namespace=%s",
		pod.Name, pod.Namespace)

	if glog.V(4) {
		glog.Infof("Got pod from file %q: %#v", filename, pod)
	} else {
		glog.V(5).Infof("Got pod from file %q: %s.%s (%s)", filename, pod.Namespace, pod.Name, pod.UID)
	}
	return pod, nil
}
