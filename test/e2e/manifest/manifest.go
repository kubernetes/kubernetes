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
	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	scheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/kubernetes/test/e2e/framework/testfiles"
)

// PodFromManifest reads a .json/yaml file and returns the pod in it.
func PodFromManifest(filename string) (*v1.Pod, error) {
	var pod v1.Pod
	data, err := testfiles.Read(filename)
	if err != nil {
		return nil, err
	}

	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), json, &pod); err != nil {
		return nil, err
	}
	return &pod, nil
}

// RcFromManifest reads a .json/yaml file and returns the rc in it.
func RcFromManifest(fileName string) (*v1.ReplicationController, error) {
	var controller v1.ReplicationController
	data, err := testfiles.Read(fileName)
	if err != nil {
		return nil, err
	}

	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), json, &controller); err != nil {
		return nil, err
	}
	return &controller, nil
}

// SvcFromManifest reads a .json/yaml file and returns the service in it.
func SvcFromManifest(fileName string) (*v1.Service, error) {
	var svc v1.Service
	data, err := testfiles.Read(fileName)
	if err != nil {
		return nil, err
	}

	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), json, &svc); err != nil {
		return nil, err
	}
	return &svc, nil
}

// StatefulSetFromManifest returns a StatefulSet from a manifest stored in fileName in the Namespace indicated by ns.
func StatefulSetFromManifest(fileName, ns string) (*appsv1.StatefulSet, error) {
	var ss appsv1.StatefulSet
	data, err := testfiles.Read(fileName)
	if err != nil {
		return nil, err
	}

	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), json, &ss); err != nil {
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
func DaemonSetFromManifest(fileName, ns string) (*appsv1.DaemonSet, error) {
	var ds appsv1.DaemonSet
	data, err := testfiles.Read(fileName)
	if err != nil {
		return nil, err
	}

	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	err = runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), json, &ds)
	if err != nil {
		return nil, err
	}
	ds.Namespace = ns
	return &ds, nil
}

// RoleFromManifest returns a Role from a manifest stored in fileName in the Namespace indicated by ns.
func RoleFromManifest(fileName, ns string) (*rbacv1.Role, error) {
	var role rbacv1.Role
	data, err := testfiles.Read(fileName)
	if err != nil {
		return nil, err
	}

	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	err = runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), json, &role)
	if err != nil {
		return nil, err
	}
	role.Namespace = ns
	return &role, nil
}
