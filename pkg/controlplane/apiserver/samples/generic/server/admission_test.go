/*
Copyright 2024 The Kubernetes Authors.

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

package server

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
	"k8s.io/kubernetes/plugin/pkg/admission/limitranger"
	"k8s.io/kubernetes/plugin/pkg/admission/network/defaultingressclass"
	"k8s.io/kubernetes/plugin/pkg/admission/nodetaint"
	"k8s.io/kubernetes/plugin/pkg/admission/podtopologylabels"
	podpriority "k8s.io/kubernetes/plugin/pkg/admission/priority"
	"k8s.io/kubernetes/plugin/pkg/admission/runtimeclass"
	"k8s.io/kubernetes/plugin/pkg/admission/security/podsecurity"
	"k8s.io/kubernetes/plugin/pkg/admission/storage/persistentvolume/resize"
	"k8s.io/kubernetes/plugin/pkg/admission/storage/storageclass/setdefault"
	"k8s.io/kubernetes/plugin/pkg/admission/storage/storageobjectinuseprotection"
)

var intentionallyOffPlugins = sets.New[string](
	limitranger.PluginName,                  // LimitRanger
	setdefault.PluginName,                   // DefaultStorageClass
	resize.PluginName,                       // PersistentVolumeClaimResize
	storageobjectinuseprotection.PluginName, // StorageObjectInUseProtection
	podpriority.PluginName,                  // Priority
	nodetaint.PluginName,                    // TaintNodesByCondition
	runtimeclass.PluginName,                 // RuntimeClass
	defaultingressclass.PluginName,          // DefaultIngressClass
	podsecurity.PluginName,                  // PodSecurity
	podtopologylabels.PluginName,            // PodTopologyLabels
)

func TestDefaultOffAdmissionPlugins(t *testing.T) {
	expectedOff := kubeoptions.DefaultOffAdmissionPlugins().Union(intentionallyOffPlugins)
	if missing := DefaultOffAdmissionPlugins().Difference(expectedOff); missing.Len() > 0 {
		t.Fatalf("generic DefaultOffAdmissionPlugins() is incomplete, double check: %v", missing)
	}
	if unexpected := expectedOff.Difference(DefaultOffAdmissionPlugins()); unexpected.Len() > 0 {
		t.Fatalf("generic DefaultOffAdmissionPlugins() has unepxeced plugins, double check: %v", unexpected)
	}
}
