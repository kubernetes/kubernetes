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

package autoscaling

import (
	"context"
	"time"

	gcm "google.golang.org/api/monitoring/v3"
	as "k8s.io/api/autoscaling/v2beta1"
	corev1 "k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/instrumentation/monitoring"

	. "github.com/onsi/ginkgo"
	"golang.org/x/oauth2/google"
)

const (
	stackdriverExporterDeployment = "stackdriver-exporter-deployment"
	dummyDeploymentName           = "dummy-deployment"
	stackdriverExporterPod        = "stackdriver-exporter-pod"
)

var _ = SIGDescribe("[HPA] Horizontal pod autoscaling (scale resource: Custom Metrics from Stackdriver)", func() {
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce", "gke")
	})

	f := framework.NewDefaultFramework("horizontal-pod-autoscaling")

	It("should scale down with Custom Metric of type Pod from Stackdriver [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 2
		scaledReplicas := 1
		// metric should cause scale down
		metricValue := int64(100)
		metricTarget := 2 * metricValue
		deployment := monitoring.SimpleStackdriverExporterDeployment(stackdriverExporterDeployment, f.Namespace.ObjectMeta.Name, int32(initialReplicas), metricValue)
		customMetricTest(f, f.ClientSet, simplePodsHPA(f.Namespace.ObjectMeta.Name, metricTarget), deployment, nil, initialReplicas, scaledReplicas)
	})

	It("should scale down with Custom Metric of type Object from Stackdriver [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 2
		scaledReplicas := 1
		// metric should cause scale down
		metricValue := int64(100)
		metricTarget := 2 * metricValue
		deployment := monitoring.SimpleStackdriverExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas), metricValue)
		pod := monitoring.StackdriverExporterPod(stackdriverExporterPod, f.Namespace.Name, stackdriverExporterPod, monitoring.CustomMetricName, metricValue)
		customMetricTest(f, f.ClientSet, objectHPA(f.Namespace.ObjectMeta.Name, metricTarget), deployment, pod, initialReplicas, scaledReplicas)
	})

	It("should scale down with Custom Metric of type Pod from Stackdriver with Prometheus [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 2
		scaledReplicas := 1
		// metric should cause scale down
		metricValue := int64(100)
		metricTarget := 2 * metricValue
		deployment := monitoring.PrometheusExporterDeployment(stackdriverExporterDeployment, f.Namespace.ObjectMeta.Name, int32(initialReplicas), metricValue)
		customMetricTest(f, f.ClientSet, simplePodsHPA(f.Namespace.ObjectMeta.Name, metricTarget), deployment, nil, initialReplicas, scaledReplicas)
	})

	It("should scale up with two metrics of type Pod from Stackdriver [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 1
		scaledReplicas := 3
		// metric 1 would cause a scale down, if not for metric 2
		metric1Value := int64(100)
		metric1Target := 2 * metric1Value
		// metric2 should cause a scale up
		metric2Value := int64(200)
		metric2Target := int64(0.5 * float64(metric2Value))
		containers := []monitoring.CustomMetricContainerSpec{
			{
				Name:        "stackdriver-exporter-metric1",
				MetricName:  "metric1",
				MetricValue: metric1Value,
			},
			{
				Name:        "stackdriver-exporter-metric2",
				MetricName:  "metric2",
				MetricValue: metric2Value,
			},
		}
		metricTargets := map[string]int64{"metric1": metric1Target, "metric2": metric2Target}
		deployment := monitoring.StackdriverExporterDeployment(stackdriverExporterDeployment, f.Namespace.ObjectMeta.Name, int32(initialReplicas), containers)
		customMetricTest(f, f.ClientSet, podsHPA(f.Namespace.ObjectMeta.Name, stackdriverExporterDeployment, metricTargets), deployment, nil, initialReplicas, scaledReplicas)
	})
})

func customMetricTest(f *framework.Framework, kubeClient clientset.Interface, hpa *as.HorizontalPodAutoscaler,
	deployment *extensions.Deployment, pod *corev1.Pod, initialReplicas, scaledReplicas int) {
	projectId := framework.TestContext.CloudConfig.ProjectID

	ctx := context.Background()
	client, err := google.DefaultClient(ctx, gcm.CloudPlatformScope)

	// Hack for running tests locally, needed to authenticate in Stackdriver
	// If this is your use case, create application default credentials:
	// $ gcloud auth application-default login
	// and uncomment following lines:
	/*
		ts, err := google.DefaultTokenSource(oauth2.NoContext)
		framework.Logf("Couldn't get application default credentials, %v", err)
		if err != nil {
			framework.Failf("Error accessing application default credentials, %v", err)
		}
		client := oauth2.NewClient(oauth2.NoContext, ts)
	*/

	gcmService, err := gcm.New(client)
	if err != nil {
		framework.Failf("Failed to create gcm service, %v", err)
	}

	// Set up a cluster: create a custom metric and set up k8s-sd adapter
	err = monitoring.CreateDescriptors(gcmService, projectId)
	if err != nil {
		framework.Failf("Failed to create metric descriptor: %v", err)
	}
	defer monitoring.CleanupDescriptors(gcmService, projectId)

	err = monitoring.CreateAdapter()
	if err != nil {
		framework.Failf("Failed to set up: %v", err)
	}
	defer monitoring.CleanupAdapter()

	// Run application that exports the metric
	err = createDeploymentToScale(f, kubeClient, deployment, pod)
	if err != nil {
		framework.Failf("Failed to create stackdriver-exporter pod: %v", err)
	}
	defer cleanupDeploymentsToScale(f, kubeClient, deployment, pod)

	// Wait for the deployment to run
	waitForReplicas(deployment.ObjectMeta.Name, f.Namespace.ObjectMeta.Name, kubeClient, 15*time.Minute, initialReplicas)

	// Autoscale the deployment
	_, err = kubeClient.AutoscalingV2beta1().HorizontalPodAutoscalers(f.Namespace.ObjectMeta.Name).Create(hpa)
	if err != nil {
		framework.Failf("Failed to create HPA: %v", err)
	}

	waitForReplicas(deployment.ObjectMeta.Name, f.Namespace.ObjectMeta.Name, kubeClient, 15*time.Minute, scaledReplicas)
}

func createDeploymentToScale(f *framework.Framework, cs clientset.Interface, deployment *extensions.Deployment, pod *corev1.Pod) error {
	if deployment != nil {
		_, err := cs.Extensions().Deployments(f.Namespace.ObjectMeta.Name).Create(deployment)
		if err != nil {
			return err
		}
	}
	if pod != nil {
		_, err := cs.CoreV1().Pods(f.Namespace.ObjectMeta.Name).Create(pod)
		if err != nil {
			return err
		}
	}
	return nil
}

func cleanupDeploymentsToScale(f *framework.Framework, cs clientset.Interface, deployment *extensions.Deployment, pod *corev1.Pod) {
	if deployment != nil {
		_ = cs.Extensions().Deployments(f.Namespace.ObjectMeta.Name).Delete(deployment.ObjectMeta.Name, &metav1.DeleteOptions{})
	}
	if pod != nil {
		_ = cs.CoreV1().Pods(f.Namespace.ObjectMeta.Name).Delete(pod.ObjectMeta.Name, &metav1.DeleteOptions{})
	}
}

func simplePodsHPA(namespace string, metricTarget int64) *as.HorizontalPodAutoscaler {
	return podsHPA(namespace, stackdriverExporterDeployment, map[string]int64{monitoring.CustomMetricName: metricTarget})
}

func podsHPA(namespace string, deploymentName string, metricTargets map[string]int64) *as.HorizontalPodAutoscaler {
	var minReplicas int32 = 1
	metrics := []as.MetricSpec{}
	for metric, target := range metricTargets {
		metrics = append(metrics, as.MetricSpec{
			Type: as.PodsMetricSourceType,
			Pods: &as.PodsMetricSource{
				MetricName:         metric,
				TargetAverageValue: *resource.NewQuantity(target, resource.DecimalSI),
			},
		})
	}
	return &as.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "custom-metrics-pods-hpa",
			Namespace: namespace,
		},
		Spec: as.HorizontalPodAutoscalerSpec{
			Metrics:     metrics,
			MaxReplicas: 3,
			MinReplicas: &minReplicas,
			ScaleTargetRef: as.CrossVersionObjectReference{
				APIVersion: "extensions/v1beta1",
				Kind:       "Deployment",
				Name:       deploymentName,
			},
		},
	}
}

func objectHPA(namespace string, metricTarget int64) *as.HorizontalPodAutoscaler {
	var minReplicas int32 = 1
	return &as.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "custom-metrics-objects-hpa",
			Namespace: namespace,
		},
		Spec: as.HorizontalPodAutoscalerSpec{
			Metrics: []as.MetricSpec{
				{
					Type: as.ObjectMetricSourceType,
					Object: &as.ObjectMetricSource{
						MetricName: monitoring.CustomMetricName,
						Target: as.CrossVersionObjectReference{
							Kind: "Pod",
							Name: stackdriverExporterPod,
						},
						TargetValue: *resource.NewQuantity(metricTarget, resource.DecimalSI),
					},
				},
			},
			MaxReplicas: 3,
			MinReplicas: &minReplicas,
			ScaleTargetRef: as.CrossVersionObjectReference{
				APIVersion: "extensions/v1beta1",
				Kind:       "Deployment",
				Name:       dummyDeploymentName,
			},
		},
	}
}

func waitForReplicas(deploymentName, namespace string, cs clientset.Interface, timeout time.Duration, desiredReplicas int) {
	interval := 20 * time.Second
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		deployment, err := cs.ExtensionsV1beta1().Deployments(namespace).Get(deploymentName, metav1.GetOptions{})
		if err != nil {
			framework.Failf("Failed to get replication controller %s: %v", deployment, err)
		}
		replicas := int(deployment.Status.ReadyReplicas)
		framework.Logf("waiting for %d replicas (current: %d)", desiredReplicas, replicas)
		return replicas == desiredReplicas, nil // Expected number of replicas found. Exit.
	})
	if err != nil {
		framework.Failf("Timeout waiting %v for %v replicas", timeout, desiredReplicas)
	}
}
