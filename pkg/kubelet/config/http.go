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
	"io/ioutil"
	"net/http"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
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
		s.updates <- kubelet.PodUpdate{[]api.Pod{}, kubelet.SET, kubelet.HTTPSource}
		return fmt.Errorf("zero-length data received from %v", s.url)
	}
	// Short circuit if the manifest has not changed since the last time it was read.
	if bytes.Compare(data, s.data) == 0 {
		return nil
	}
	s.data = data

	// First try as if it's a single manifest
	parsed, manifest, pod, singleErr := tryDecodeSingle(data)
	if parsed {
		if singleErr != nil {
			// It parsed but could not be used.
			return singleErr
		}
		// It parsed!
		if err = applyDefaults(&pod, s.url); err != nil {
			return err
		}
		s.updates <- kubelet.PodUpdate{[]api.Pod{pod}, kubelet.SET, kubelet.HTTPSource}
		return nil
	}

	// That didn't work, so try an array of manifests.
	parsed, manifests, pods, multiErr := tryDecodeList(data)
	if parsed {
		if multiErr != nil {
			// It parsed but could not be used.
			return multiErr
		}
		// A single manifest that did not pass semantic validation will yield an empty
		// array of manifests (and no error) when unmarshaled as such.  In that case,
		// if the single manifest at least had a Version, we return the single-manifest
		// error (if any).
		if len(manifests) == 0 && len(manifest.Version) != 0 {
			return singleErr
		}
		// Assume it parsed.
		for i := range pods.Items {
			pod := &pods.Items[i]
			if err = applyDefaults(pod, s.url); err != nil {
				return err
			}
		}
		s.updates <- kubelet.PodUpdate{pods.Items, kubelet.SET, kubelet.HTTPSource}
		return nil
	}

	return fmt.Errorf("%v: received '%v', but couldn't parse as a "+
		"single manifest (%v: %+v) or as multiple manifests (%v: %+v).\n",
		s.url, string(data), singleErr, manifest, multiErr, manifests)
}

func tryDecodeSingle(data []byte) (parsed bool, manifest v1beta1.ContainerManifest, pod api.Pod, err error) {
	// TODO: should be api.Scheme.Decode
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
	if err = yaml.Unmarshal(data, &manifest); err != nil {
		return false, manifest, pod, err
	}
	newManifest := api.ContainerManifest{}
	if err = api.Scheme.Convert(&manifest, &newManifest); err != nil {
		return false, manifest, pod, err
	}
	if errs := validation.ValidateManifest(&newManifest); len(errs) > 0 {
		err = fmt.Errorf("invalid manifest: %v", errs)
		return false, manifest, pod, err
	}
	if err = api.Scheme.Convert(&newManifest, &pod); err != nil {
		return true, manifest, pod, err
	}
	// Success.
	return true, manifest, pod, nil
}

func tryDecodeList(data []byte) (parsed bool, manifests []v1beta1.ContainerManifest, pods api.PodList, err error) {
	// TODO: should be api.Scheme.Decode
	// See the comment in tryDecodeSingle().
	if err = yaml.Unmarshal(data, &manifests); err != nil {
		return false, manifests, pods, err
	}
	newManifests := []api.ContainerManifest{}
	if err = api.Scheme.Convert(&manifests, &newManifests); err != nil {
		return false, manifests, pods, err
	}
	for i := range newManifests {
		manifest := &newManifests[i]
		if errs := validation.ValidateManifest(manifest); len(errs) > 0 {
			err = fmt.Errorf("invalid manifest: %v", errs)
			return false, manifests, pods, err
		}
	}
	list := api.ContainerManifestList{Items: newManifests}
	if err = api.Scheme.Convert(&list, &pods); err != nil {
		return true, manifests, pods, err
	}
	// Success.
	return true, manifests, pods, nil
}

func applyDefaults(pod *api.Pod, url string) error {
	if len(pod.UID) == 0 {
		hasher := md5.New()
		fmt.Fprintf(hasher, "url:%s", url)
		util.DeepHashObject(hasher, pod)
		pod.UID = types.UID(hex.EncodeToString(hasher.Sum(nil)[0:]))
		glog.V(5).Infof("Generated UID %q for pod %q from URL %s", pod.UID, pod.Name, url)
	}
	// This is required for backward compatibility, and should be removed once we
	// completely deprecate ContainerManifest.
	var err error
	if len(pod.Name) == 0 {
		pod.Name = string(pod.UID)
	}
	pod.Name, err = GeneratePodName(pod.Name)
	if err != nil {
		return err
	}
	glog.V(5).Infof("Generated Name %q for UID %q from URL %s", pod.Name, pod.UID, url)

	// Always overrides the namespace.
	pod.Namespace = kubelet.NamespaceDefault
	glog.V(5).Infof("Using namespace %q for pod %q from URL %s", pod.Namespace, pod.Name, url)
	return nil
}
