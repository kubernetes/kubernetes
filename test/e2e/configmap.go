/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
)

var _ = Describe("ConfigMap", func() {
	f := NewFramework("configmap")

	It("should be consumable via environment variable [Conformance]", func() {
		name := "configmap-test-" + string(util.NewUUID())
		configMap := &api.ConfigMap{
			ObjectMeta: api.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      name,
			},
			Data: map[string]string{
				"data-1": "value-1",
				"data-2": "value-2",
				"data-3": "value-3",
			},
		}

		By(fmt.Sprintf("Creating configMap %v/%v", f.Namespace.Name, configMap.Name))
		defer func() {
			By("Cleaning up the configMap")
			if err := f.Client.ConfigMaps(f.Namespace.Name).Delete(configMap.Name); err != nil {
				Failf("unable to delete configMap %v: %v", configMap.Name, err)
			}
		}()
		var err error
		if configMap, err = f.Client.ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
			Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: "pod-configmaps-" + string(util.NewUUID()),
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:    "env-test",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"sh", "-c", "env"},
						Env: []api.EnvVar{
							{
								Name: "CONFIG_DATA_1",
								ValueFrom: &api.EnvVarSource{
									ConfigMapKeyRef: &api.ConfigMapKeySelector{
										LocalObjectReference: api.LocalObjectReference{
											Name: name,
										},
										Key: "data-1",
									},
								},
							},
						},
					},
				},
				RestartPolicy: api.RestartPolicyNever,
			},
		}

		testContainerOutput("consume configMaps", f.Client, pod, 0, []string{
			"CONFIG_DATA_1=value-1",
		}, f.Namespace.Name)
	})
})
