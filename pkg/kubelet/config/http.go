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

// Reads the pod configuration from an HTTP GET response.
package config

import (
	"bytes"
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"hash/adler32"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/ghodss/yaml"
	"github.com/golang/glog"
)

type sourceURL struct {
	url     string
	updates chan<- interface{}
	data    []byte
}

func NewSourceURL(url string, period time.Duration, updates chan<- interface{}) {
	config := &sourceURL{
		url:     url,
		updates: updates,
		data:    nil,
	}
	glog.V(1).Infof("Watching URL %s", url)
	go util.Forever(config.run, period)
}

func (s *sourceURL) run() {
	if err := s.extractFromURL(); err != nil {
		glog.Errorf("Failed to read URL: %v", err)
	}
}

func (s *sourceURL) extractFromURL() error {
	resp, err := http.Get(s.url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("%v: %v", s.url, resp.Status)
	}
	if len(data) == 0 {
		// Emit an update with an empty PodList to allow HTTPSource to be marked as seen
		s.updates <- kubelet.PodUpdate{[]api.BoundPod{}, kubelet.SET, kubelet.HTTPSource}
		return fmt.Errorf("zero-length data received from %v", s.url)
	}
	// Short circuit if the manifest has not changed since the last time it was read.
	if bytes.Compare(data, s.data) == 0 {
		return nil
	}
	s.data = data

	// First try as if it's a single manifest
	var manifest api.ContainerManifest
	// TODO: should be api.Scheme.Decode
	singleErr := yaml.Unmarshal(data, &manifest)
	if singleErr == nil {
		if errs := validation.ValidateManifest(&manifest); len(errs) > 0 {
			singleErr = fmt.Errorf("invalid manifest: %v", errs)
		}
	}
	if singleErr == nil {
		pod := api.BoundPod{}
		if err := api.Scheme.Convert(&manifest, &pod); err != nil {
			return err
		}
		applyDefaults(&pod, s.url)
		s.updates <- kubelet.PodUpdate{[]api.BoundPod{pod}, kubelet.SET, kubelet.HTTPSource}
		return nil
	}

	// That didn't work, so try an array of manifests.
	var manifests []api.ContainerManifest
	// TODO: should be api.Scheme.Decode
	multiErr := yaml.Unmarshal(data, &manifests)
	// We're not sure if the person reading the logs is going to care about the single or
	// multiple manifest unmarshalling attempt, so we need to put both in the logs, as is
	// done at the end. Hence not returning early here.
	if multiErr == nil {
		for _, manifest := range manifests {
			if errs := validation.ValidateManifest(&manifest); len(errs) > 0 {
				multiErr = fmt.Errorf("invalid manifest: %v", errs)
				break
			}
		}
	}
	if multiErr == nil {
		// A single manifest that did not pass semantic validation will yield an empty
		// array of manifests (and no error) when unmarshaled as such.  In that case,
		// if the single manifest at least had a Version, we return the single-manifest
		// error (if any).
		if len(manifests) == 0 && len(manifest.Version) != 0 {
			return singleErr
		}
		list := api.ContainerManifestList{Items: manifests}
		boundPods := &api.BoundPods{}
		if err := api.Scheme.Convert(&list, boundPods); err != nil {
			return err
		}
		for i := range boundPods.Items {
			pod := &boundPods.Items[i]
			applyDefaults(pod, s.url)
		}
		s.updates <- kubelet.PodUpdate{boundPods.Items, kubelet.SET, kubelet.HTTPSource}
		return nil
	}

	return fmt.Errorf("%v: received '%v', but couldn't parse as a "+
		"single manifest (%v: %+v) or as multiple manifests (%v: %+v).\n",
		s.url, string(data), singleErr, manifest, multiErr, manifests)
}

func applyDefaults(pod *api.BoundPod, url string) {
	if len(pod.UID) == 0 {
		hasher := md5.New()
		fmt.Fprintf(hasher, "url:%s", url)
		util.DeepHashObject(hasher, pod)
		pod.UID = hex.EncodeToString(hasher.Sum(nil)[0:])
		glog.V(5).Infof("Generated UID %q for pod %q from URL %s", pod.UID, pod.Name, url)
	}
	if len(pod.Namespace) == 0 {
		hasher := adler32.New()
		fmt.Fprint(hasher, url)
		pod.Namespace = fmt.Sprintf("url-%08x", hasher.Sum32())
		glog.V(5).Infof("Generated namespace %q for pod %q from URL %s", pod.Namespace, pod.Name, url)
	}
}
