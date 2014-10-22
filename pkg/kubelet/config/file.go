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
	"crypto/sha1"
	"encoding/base32"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/golang/glog"
	"gopkg.in/v1/yaml"
)

type SourceFile struct {
	path    string
	updates chan<- interface{}
}

func NewSourceFile(path string, period time.Duration, updates chan<- interface{}) *SourceFile {
	config := &SourceFile{
		path:    path,
		updates: updates,
	}
	glog.V(1).Infof("Watching path %q", path)
	go util.Forever(config.run, period)
	return config
}

func (s *SourceFile) run() {
	if err := s.extractFromPath(); err != nil {
		glog.Errorf("Unable to read config path %q: %v", s.path, err)
	}
}

func (s *SourceFile) extractFromPath() error {
	path := s.path
	statInfo, err := os.Stat(path)
	if err != nil {
		if !os.IsNotExist(err) {
			return err
		}
		return fmt.Errorf("path does not exist, ignoring")
	}

	switch {
	case statInfo.Mode().IsDir():
		pods, err := extractFromDir(path)
		if err != nil {
			return err
		}
		s.updates <- kubelet.PodUpdate{pods, kubelet.SET}

	case statInfo.Mode().IsRegular():
		pod, err := extractFromFile(path)
		if err != nil {
			return err
		}
		s.updates <- kubelet.PodUpdate{[]api.BoundPod{pod}, kubelet.SET}

	default:
		return fmt.Errorf("path is not a directory or file")
	}

	return nil
}

// Get as many pod configs as we can from a directory.  Return an error iff something
// prevented us from reading anything at all.  Do not return an error if only some files
// were problematic.
func extractFromDir(name string) ([]api.BoundPod, error) {
	dirents, err := filepath.Glob(filepath.Join(name, "[^.]*"))
	if err != nil {
		return nil, fmt.Errorf("glob failed: %v", err)
	}
	if len(dirents) == 0 {
		return nil, nil
	}

	sort.Strings(dirents)
	pods := []api.BoundPod{}
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

func extractFromFile(filename string) (api.BoundPod, error) {
	var pod api.BoundPod

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

	manifest := &api.ContainerManifest{}
	// TODO: use api.Scheme.DecodeInto
	if err := yaml.Unmarshal(data, manifest); err != nil {
		return pod, fmt.Errorf("can't unmarshal file %q: %v", filename, err)
	}

	if err := api.Scheme.Convert(manifest, &pod); err != nil {
		return pod, fmt.Errorf("can't convert pod from file %q: %v", filename, err)
	}

	pod.Name = simpleSubdomainSafeHash(filename)
	if len(pod.UID) == 0 {
		pod.UID = simpleSubdomainSafeHash(filename)
	}
	if len(pod.Namespace) == 0 {
		pod.Namespace = api.NamespaceDefault
	}

	if glog.V(4) {
		glog.Infof("Got pod from file %q: %#v", filename, pod)
	} else {
		glog.V(1).Infof("Got pod from file %q: %s.%s (%s)", filename, pod.Namespace, pod.Name, pod.UID)
	}
	return pod, nil
}

var simpleSubdomainSafeEncoding = base32.NewEncoding("0123456789abcdefghijklmnopqrstuv")
var unsafeDNSLabelReplacement = regexp.MustCompile("[^a-z0-9]+")

// simpleSubdomainSafeHash generates a pod name for the given path that is
// suitable as a subdomain label.
func simpleSubdomainSafeHash(path string) string {
	name := strings.ToLower(filepath.Base(path))
	name = unsafeDNSLabelReplacement.ReplaceAllString(name, "")
	hasher := sha1.New()
	hasher.Write([]byte(path))
	sha := simpleSubdomainSafeEncoding.EncodeToString(hasher.Sum(nil))
	return fmt.Sprintf("%.15s%.30s", name, sha)
}
