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

package storage

import (
	"context"

	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2econformance "k8s.io/kubernetes/test/e2e/framework/conformance"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

var _ = utils.SIGDescribe("VolumeAttributesClass", framework.WithFeatureGate(features.VolumeAttributesClass), func() {
	f := framework.NewDefaultFramework("volumeattributesclass-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelRestricted

	/*
	   Release: v1.35
	   Testname: CRUD operations for VolumeAttributesClasses
	   Description: kube-apiserver must support create/get/list/update/patch/delete operations for storage.k8s.io/v1 VolumeAttributesClass.
	*/
	framework.ConformanceIt("VolumeAttributesClass API availability", func(ctx context.Context) {
		e2econformance.TestResource(ctx, f,
			&e2econformance.ResourceTestcase[*storagev1.VolumeAttributesClass]{
				GVR:        storagev1.SchemeGroupVersion.WithResource("volumeattributesclasses"),
				Namespaced: ptr.To(false),
				InitialSpec: &storagev1.VolumeAttributesClass{
					DriverName: "e2e-fake-csi-driver",
					Parameters: map[string]string{"foo": "bar"},
				},
				UpdateSpec: func(obj *storagev1.VolumeAttributesClass) *storagev1.VolumeAttributesClass {
					metav1.SetMetaDataLabel(&obj.ObjectMeta, "foo", "bar")
					return obj
				},
				StrategicMergePatchSpec: `{"metadata": {"labels": {"foo": "bar"}}}`,
			},
		)
	})
})
