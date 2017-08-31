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

package monitoring

import (
	"fmt"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var (
	CustomMetricName = "foo-metric"
	UnusedMetricName = "unused-metric"
	MetricValue1     = int64(448)
	MetricValue2     = int64(446)

	SDExporterPod1 = &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "sd-exporter-1",
			Namespace: "default",
			Labels: map[string]string{
				"name": "sd-exporter",
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "sd-exporter",
					Image:           "gcr.io/google-containers/sd-dummy-exporter:v0.1.0",
					ImagePullPolicy: v1.PullPolicy("Always"),
					Command:         []string{"/sd_dummy_exporter", "--pod-id=$(POD_ID)", "--metric-name=" + CustomMetricName, fmt.Sprintf("--metric-value=%v", MetricValue1)},
					Env: []v1.EnvVar{
						{
							Name: "POD_ID",
							ValueFrom: &v1.EnvVarSource{
								FieldRef: &v1.ObjectFieldSelector{
									FieldPath: "metadata.uid",
								},
							},
						},
					},
					Ports: []v1.ContainerPort{{ContainerPort: 80}},
				},
			},
		},
	}
	SDExporterPod2 = &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "sd-exporter-2",
			Namespace: "default",
			Labels: map[string]string{
				"name": "sd-exporter",
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "sd-exporter",
					Image:           "gcr.io/google-containers/sd-dummy-exporter:v0.1.0",
					ImagePullPolicy: v1.PullPolicy("Always"),
					Command:         []string{"/sd_dummy_exporter", "--pod-id=$(POD_ID)", "--metric-name=" + CustomMetricName, fmt.Sprintf("--metric-value=%v", MetricValue2)},
					Env: []v1.EnvVar{
						{
							Name: "POD_ID",
							ValueFrom: &v1.EnvVarSource{
								FieldRef: &v1.ObjectFieldSelector{
									FieldPath: "metadata.uid",
								},
							},
						},
					},
					Ports: []v1.ContainerPort{{ContainerPort: 80}},
				},
			},
		},
	}
)
