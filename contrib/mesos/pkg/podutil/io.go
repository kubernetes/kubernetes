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

package podutil

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/runtime"
	utilyaml "k8s.io/kubernetes/pkg/util/yaml"
)

func WriteToDir(pods <-chan *api.Pod, destDir string) error {
	err := os.MkdirAll(destDir, 0660)
	if err != nil {
		return err
	}
	for p := range pods {
		filename, ok := p.Annotations[meta.StaticPodFilenameKey]
		if !ok {
			log.Warningf("skipping static pod %s/%s that had no filename", p.Namespace, p.Name)
			continue
		}
		raw, err := runtime.Encode(api.Codecs.LegacyCodec(v1.SchemeGroupVersion), p)
		if err != nil {
			log.Errorf("failed to encode static pod as v1 object: %v", err)
			continue
		}
		destfile := filepath.Join(destDir, filename)
		err = ioutil.WriteFile(destfile, raw, 0660)
		if err != nil {
			log.Errorf("failed to write static pod file %q: %v", destfile, err)
		}
		log.V(1).Infof("wrote static pod %s/%s to %s", p.Namespace, p.Name, destfile)
	}
	return nil
}

func ReadFromDir(dirpath string) (<-chan *api.Pod, <-chan error) {
	pods := make(chan *api.Pod)
	errors := make(chan error)
	go func() {
		defer close(pods)
		defer close(errors)
		files, err := ioutil.ReadDir(dirpath)
		if err != nil {
			errors <- fmt.Errorf("error scanning static pods directory: %q: %v", dirpath, err)
			return
		}
		for _, f := range files {
			if f.IsDir() || f.Size() == 0 {
				continue
			}
			filename := filepath.Join(dirpath, f.Name())
			log.V(1).Infof("reading static pod conf from file %q", filename)

			data, err := ioutil.ReadFile(filename)
			if err != nil {
				errors <- fmt.Errorf("failed to read static pod file: %q: %v", filename, err)
				continue
			}

			parsed, pod, err := tryDecodeSinglePod(data)
			if !parsed {
				if err != nil {
					errors <- fmt.Errorf("error parsing static pod file %q: %v", filename, err)
				}
				continue
			}
			if err != nil {
				errors <- fmt.Errorf("error validating static pod file %q: %v", filename, err)
				continue
			}
			Annotate(&pod.ObjectMeta, map[string]string{meta.StaticPodFilenameKey: f.Name()})
			pods <- pod
		}
	}()
	return pods, errors
}

// tryDecodeSinglePod was copied from pkg/kubelet/config/common.go v1.0.5
func tryDecodeSinglePod(data []byte) (parsed bool, pod *api.Pod, err error) {
	// JSON is valid YAML, so this should work for everything.
	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return false, nil, err
	}
	obj, err := runtime.Decode(api.Codecs.UniversalDecoder(), json)
	if err != nil {
		return false, pod, err
	}
	// Check whether the object could be converted to single pod.
	if _, ok := obj.(*api.Pod); !ok {
		err = fmt.Errorf("invalid pod: %+v", obj)
		return false, pod, err
	}
	newPod := obj.(*api.Pod)
	if errs := validation.ValidatePod(newPod); len(errs) > 0 {
		err = fmt.Errorf("invalid pod: %v", errs)
		return true, pod, err
	}
	return true, newPod, nil
}
