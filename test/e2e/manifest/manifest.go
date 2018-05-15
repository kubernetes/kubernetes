/*
Copyright 2017 The Kubernetes Authors.

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

package manifest

import (
	"fmt"
	"io/ioutil"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/test/e2e/generated"
)

// PodFromManifest reads a .json/yaml file and returns the pod in it.
func PodFromManifest(filename string) (*v1.Pod, error) {
	var pod v1.Pod
	data := generated.ReadOrDie(filename)

	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	if err := runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), json, &pod); err != nil {
		return nil, err
	}
	return &pod, nil
}

// RcFromManifest reads a .json/yaml file and returns the rc in it.
func RcFromManifest(fileName string) (*v1.ReplicationController, error) {
	var controller v1.ReplicationController
	data := generated.ReadOrDie(fileName)

	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	if err := runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), json, &controller); err != nil {
		return nil, err
	}
	return &controller, nil
}

// SvcFromManifest reads a .json/yaml file and returns the service in it.
func SvcFromManifest(fileName string) (*v1.Service, error) {
	var svc v1.Service
	data := generated.ReadOrDie(fileName)

	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	if err := runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), json, &svc); err != nil {
		return nil, err
	}
	return &svc, nil
}

// IngressFromManifest reads a .json/yaml file and returns the ingress in it.
func IngressFromManifest(fileName string) (*extensions.Ingress, error) {
	var ing extensions.Ingress
	data := generated.ReadOrDie(fileName)

	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	if err := runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), json, &ing); err != nil {
		return nil, err
	}
	return &ing, nil
}

// IngressToManifest generates a yaml file in the given path with the given ingress.
// Assumes that a directory exists at the given path.
func IngressToManifest(ing *extensions.Ingress, path string) error {
	serialized, err := util.MarshalToYaml(ing, extensions.SchemeGroupVersion)
	if err != nil {
		return fmt.Errorf("failed to marshal ingress %v to YAML: %v", ing, err)
	}

	if err := ioutil.WriteFile(path, serialized, 0600); err != nil {
		return fmt.Errorf("error in writing ingress to file: %s", err)
	}
	return nil
}

// StatefulSetFromManifest returns a StatefulSet from a manifest stored in fileName in the Namespace indicated by ns.
func StatefulSetFromManifest(fileName, ns string) (*apps.StatefulSet, error) {
	var ss apps.StatefulSet
	data := generated.ReadOrDie(fileName)

	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	if err := runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), json, &ss); err != nil {
		return nil, err
	}
	ss.Namespace = ns
	if ss.Spec.Selector == nil {
		ss.Spec.Selector = &metav1.LabelSelector{
			MatchLabels: ss.Spec.Template.Labels,
		}
	}
	return &ss, nil
}

// DaemonSetFromManifest returns a DaemonSet from a manifest stored in fileName in the Namespace indicated by ns.
func DaemonSetFromManifest(fileName, ns string) (*apps.DaemonSet, error) {
	var ds apps.DaemonSet
	data := generated.ReadOrDie(fileName)

	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	err = runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), json, &ds)
	if err != nil {
		return nil, err
	}
	ds.Namespace = ns
	return &ds, nil
}
