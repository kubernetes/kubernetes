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

// Reads the pod configuration from file or a directory of files
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
	glog.Infof("Watching file %s", path)
	go util.Forever(config.run, period)
	return config
}

func (s *SourceFile) run() {
	if err := s.extractFromPath(); err != nil {
		glog.Errorf("Unable to read config file: %s", err)
	}
}

func (s *SourceFile) extractFromPath() error {
	path := s.path
	statInfo, err := os.Stat(path)
	if err != nil {
		if !os.IsNotExist(err) {
			return fmt.Errorf("unable to access path: %s", err)
		}
		return fmt.Errorf("path does not exist: %s", path)
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
		s.updates <- kubelet.PodUpdate{[]kubelet.Pod{pod}, kubelet.SET}

	default:
		return fmt.Errorf("path is not a directory or file")
	}

	return nil
}

func extractFromDir(name string) ([]kubelet.Pod, error) {
	pods := []kubelet.Pod{}

	files, err := filepath.Glob(filepath.Join(name, "[^.]*"))
	if err != nil {
		return pods, err
	}

	sort.Strings(files)

	for _, file := range files {
		pod, err := extractFromFile(file)
		if err != nil {
			return []kubelet.Pod{}, err
		}
		pods = append(pods, pod)
	}
	return pods, nil
}

func extractFromFile(name string) (kubelet.Pod, error) {
	var pod kubelet.Pod

	file, err := os.Open(name)
	if err != nil {
		return pod, err
	}
	defer file.Close()

	data, err := ioutil.ReadAll(file)
	if err != nil {
		glog.Errorf("Couldn't read from file: %v", err)
		return pod, err
	}

	if err := yaml.Unmarshal(data, &pod.Manifest); err != nil {
		return pod, fmt.Errorf("could not unmarshal manifest: %v", err)
	}

	podName := pod.Manifest.ID
	if podName == "" {
		podName = simpleSubdomainSafeHash(name)
	}
	pod.Name = podName

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
