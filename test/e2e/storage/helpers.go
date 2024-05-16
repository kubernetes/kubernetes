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
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	v12 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/client/conditions"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
)

func newStorageClass(t testsuites.StorageClassTest, ns string, prefix string) *storagev1.StorageClass {
	pluginName := t.Provisioner
	if pluginName == "" {
		pluginName = getDefaultPluginName()
	}
	if prefix == "" {
		prefix = "sc"
	}
	bindingMode := storagev1.VolumeBindingImmediate
	if t.DelayBinding {
		bindingMode = storagev1.VolumeBindingWaitForFirstConsumer
	}
	if t.Parameters == nil {
		t.Parameters = make(map[string]string)
	}

	if framework.NodeOSDistroIs("windows") {
		// fstype might be forced from outside, in that case skip setting a default
		if _, exists := t.Parameters["fstype"]; !exists {
			t.Parameters["fstype"] = e2epv.GetDefaultFSType()
			framework.Logf("settings a default fsType=%s in the storage class", t.Parameters["fstype"])
		}
	}

	sc := getStorageClass(pluginName, t.Parameters, &bindingMode, t.MountOptions, ns, prefix)
	if t.AllowVolumeExpansion {
		sc.AllowVolumeExpansion = &t.AllowVolumeExpansion
	}
	return sc
}

func getDefaultPluginName() string {
	switch {
	case framework.ProviderIs("gke"), framework.ProviderIs("gce"):
		return "kubernetes.io/gce-pd"
	case framework.ProviderIs("aws"):
		return "kubernetes.io/aws-ebs"
	case framework.ProviderIs("openstack"):
		return "kubernetes.io/cinder"
	case framework.ProviderIs("vsphere"):
		return "kubernetes.io/vsphere-volume"
	case framework.ProviderIs("azure"):
		return "kubernetes.io/azure-disk"
	}
	return ""
}

func getStorageClass(
	provisioner string,
	parameters map[string]string,
	bindingMode *storagev1.VolumeBindingMode,
	mountOptions []string,
	ns string,
	prefix string,
) *storagev1.StorageClass {
	if bindingMode == nil {
		defaultBindingMode := storagev1.VolumeBindingImmediate
		bindingMode = &defaultBindingMode
	}
	return &storagev1.StorageClass{
		TypeMeta: v12.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: v12.ObjectMeta{
			// Name must be unique, so let's base it on namespace name and the prefix (the prefix is test specific)
			GenerateName: ns + "-" + prefix,
		},
		Provisioner:       provisioner,
		Parameters:        parameters,
		VolumeBindingMode: bindingMode,
		MountOptions:      mountOptions,
	}
}

func waitForDeploymentToRecreatePod(ctx context.Context, client kubernetes.Interface, deployment *appsv1.Deployment) (v1.Pod, error) {
	var runningPod v1.Pod
	waitErr := wait.PollImmediate(10*time.Second, 5*time.Minute, func() (bool, error) {
		podList, err := e2edeployment.GetPodsForDeployment(ctx, client, deployment)
		if err != nil {
			return false, fmt.Errorf("failed to get pods for deployment: %w", err)
		}
		for _, pod := range podList.Items {
			switch pod.Status.Phase {
			case v1.PodRunning:
				runningPod = pod
				return true, nil
			case v1.PodFailed, v1.PodSucceeded:
				return false, conditions.ErrPodCompleted
			}
		}
		return false, nil
	})
	if waitErr != nil {
		return runningPod, fmt.Errorf("error waiting for recreated pod: %v", waitErr)
	}
	return runningPod, nil
}
