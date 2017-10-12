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
	gcm "google.golang.org/api/monitoring/v3"
	corev1 "k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
)

var (
	CustomMetricName  = "foo-metric"
	UnusedMetricName  = "unused-metric"
	CustomMetricValue = int64(448)
	UnusedMetricValue = int64(446)
)

// StackdriverExporterDeployment is a Deployment of simple application that exports a metric of
// fixed value to Stackdriver in a loop.
func StackdriverExporterDeployment(name, namespace string, replicas int32, metricValue int64) *extensions.Deployment {
	return &extensions.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: extensions.DeploymentSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"name": name},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"name": name,
					},
				},
				Spec: stackdriverExporterPodSpec(CustomMetricName, metricValue),
			},
			Replicas: &replicas,
		},
	}
}

// StackdriverExporterPod is a Pod of simple application that exports a metric of fixed value to
// Stackdriver in a loop.
func StackdriverExporterPod(podName, namespace, podLabel, metricName string, metricValue int64) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
			Labels: map[string]string{
				"name": podLabel,
			},
		},
		Spec: stackdriverExporterPodSpec(metricName, metricValue),
	}
}

func stackdriverExporterPodSpec(metricName string, metricValue int64) corev1.PodSpec {
	return corev1.PodSpec{
		Containers: []corev1.Container{
			{
				Name:            "stackdriver-exporter",
				Image:           "gcr.io/google-containers/sd-dummy-exporter:v0.1.0",
				ImagePullPolicy: corev1.PullPolicy("Always"),
				Command:         []string{"/sd_dummy_exporter", "--pod-id=$(POD_ID)", "--metric-name=" + metricName, fmt.Sprintf("--metric-value=%v", metricValue)},
				Env: []corev1.EnvVar{
					{
						Name: "POD_ID",
						ValueFrom: &corev1.EnvVarSource{
							FieldRef: &corev1.ObjectFieldSelector{
								FieldPath: "metadata.uid",
							},
						},
					},
				},
				Ports: []corev1.ContainerPort{{ContainerPort: 80}},
			},
		},
	}
}

// CreateAdapter creates Custom Metrics - Stackdriver adapter.
func CreateAdapter() error {
	stat, err := framework.RunKubectl("create", "-f", "https://raw.githubusercontent.com/GoogleCloudPlatform/k8s-stackdriver/master/custom-metrics-stackdriver-adapter/adapter-beta.yaml")
	framework.Logf(stat)
	return err
}

// CreateDescriptors creates descriptors for metrics: CustomMetricName and UnusedMetricName.
func CreateDescriptors(service *gcm.Service, projectId string) error {
	_, err := service.Projects.MetricDescriptors.Create(fmt.Sprintf("projects/%s", projectId), &gcm.MetricDescriptor{
		Name:       CustomMetricName,
		ValueType:  "INT64",
		Type:       "custom.googleapis.com/" + CustomMetricName,
		MetricKind: "GAUGE",
	}).Do()
	if err != nil {
		return err
	}
	_, err = service.Projects.MetricDescriptors.Create(fmt.Sprintf("projects/%s", projectId), &gcm.MetricDescriptor{
		Name:       UnusedMetricName,
		ValueType:  "INT64",
		Type:       "custom.googleapis.com/" + UnusedMetricName,
		MetricKind: "GAUGE",
	}).Do()
	return err
}

// CleanupDescriptors deletes descriptors for metrics: CustomMetricName and UnusedMetricName.
// TODO: Cleanup time series as well
func CleanupDescriptors(service *gcm.Service, projectId string) {
	_, err := service.Projects.MetricDescriptors.Delete(fmt.Sprintf("projects/%s/metricDescriptors/custom.googleapis.com/%s", projectId, CustomMetricName)).Do()
	if err != nil {
		framework.Logf("Failed to delete descriptor for metric '%s': %v", CustomMetricName, err)
	}
	_, err = service.Projects.MetricDescriptors.Delete(fmt.Sprintf("projects/%s/metricDescriptors/custom.googleapis.com/%s", projectId, UnusedMetricName)).Do()
	if err != nil {
		framework.Logf("Failed to delete descriptor for metric '%s': %v", CustomMetricName, err)
	}
}

// CleanupAdapter deletes Custom Metrics - Stackdriver adapter deployments.
func CleanupAdapter() error {
	stat, err := framework.RunKubectl("delete", "-f", "https://raw.githubusercontent.com/GoogleCloudPlatform/k8s-stackdriver/master/custom-metrics-stackdriver-adapter/adapter-beta.yaml")
	framework.Logf(stat)
	return err
}
