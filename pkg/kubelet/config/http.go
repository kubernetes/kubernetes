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

// Reads the pod configuration from an HTTP GET response
package config

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
	"gopkg.in/v1/yaml"
)

type SourceURL struct {
	url     string
	updates chan<- interface{}
}

func NewSourceURL(url string, period time.Duration, updates chan<- interface{}) *SourceURL {
	config := &SourceURL{
		url:     url,
		updates: updates,
	}
	glog.Infof("Watching URL %s", url)
	go util.Forever(config.run, period)
	return config
}

func (s *SourceURL) run() {
	if err := s.extractFromURL(); err != nil {
		glog.Errorf("Failed to read URL: %s", err)
	}
}

func (s *SourceURL) extractFromURL() error {
	resp, err := http.Get(s.url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if len(data) == 0 {
		return fmt.Errorf("zero-length data received from %v", s.url)
	}

	// First try as if it's a single manifest
	var manifest api.ContainerManifest
	singleErr := yaml.Unmarshal(data, &manifest)
	if singleErr == nil {
		if errs := api.ValidateManifest(&manifest); len(errs) > 0 {
			singleErr = fmt.Errorf("invalid manifest: %v", errs)
		}
	}
	if singleErr == nil {
		pod := kubelet.Pod{Name: manifest.ID, Manifest: manifest}
		if pod.Name == "" {
			pod.Name = "1"
		}
		s.updates <- kubelet.PodUpdate{[]kubelet.Pod{pod}, kubelet.SET}
		return nil
	}

	// That didn't work, so try an array of manifests.
	var manifests []api.ContainerManifest
	multiErr := yaml.Unmarshal(data, &manifests)
	// We're not sure if the person reading the logs is going to care about the single or
	// multiple manifest unmarshalling attempt, so we need to put both in the logs, as is
	// done at the end. Hence not returning early here.
	if multiErr == nil {
		for _, manifest := range manifests {
			if errs := api.ValidateManifest(&manifest); len(errs) > 0 {
				multiErr = fmt.Errorf("invalid manifest: %v", errs)
				break
			}
		}
	}
	if multiErr == nil {
		pods := []kubelet.Pod{}
		for i, manifest := range manifests {
			pod := kubelet.Pod{Name: manifest.ID, Manifest: manifest}
			if pod.Name == "" {
				pod.Name = fmt.Sprintf("%d", i+1)
			}
			pods = append(pods, pod)
		}
		s.updates <- kubelet.PodUpdate{pods, kubelet.SET}
		return nil
	}

	return fmt.Errorf("%v: received '%v', but couldn't parse as a "+
		"single manifest (%v: %+v) or as multiple manifests (%v: %+v).\n",
		s.url, string(data), singleErr, manifest, multiErr, manifests)
}
