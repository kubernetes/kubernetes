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

	"golang.org/x/oauth2/google"
	clientset "k8s.io/client-go/kubernetes"

	. "github.com/onsi/ginkgo"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	gcm "google.golang.org/api/monitoring/v3"
	as "k8s.io/api/autoscaling/v2beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/instrumentation/monitoring"
)

const (
	stackdriverExporterDeployment = "stackdriver-exporter-deployment"
	dummyDeploymentName           = "dummy-deployment"
	stackdriverExporterPod        = "stackdriver-exporter-pod"
)

var _ = SIGDescribe("[HPA] Horizontal pod autoscaling (scale resource: Custom Metrics from Stackdriver)", func() {
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce")
	})

	f := framework.NewDefaultFramework("horizontal-pod-autoscaling")
	var kubeClient clientset.Interface

	It("should autoscale with Custom Metrics from Stackdriver [Feature:CustomMetricsAutoscaling]", func() {
		kubeClient = f.ClientSet
		testHPA(f, kubeClient)
	})
})

func testHPA(f *framework.Framework, kubeClient clientset.Interface) {
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
	err = createDeploymentsToScale(f, kubeClient)
	if err != nil {
		framework.Failf("Failed to create stackdriver-exporter pod: %v", err)
	}
	defer cleanupDeploymentsToScale(f, kubeClient)

	// Autoscale the deployments
	err = createPodsHPA(f, kubeClient)
	if err != nil {
		framework.Failf("Failed to create 'Pods' HPA: %v", err)
	}
	err = createObjectHPA(f, kubeClient)
	if err != nil {
		framework.Failf("Failed to create 'Objects' HPA: %v", err)
	}

	waitForReplicas(stackdriverExporterDeployment, f.Namespace.ObjectMeta.Name, kubeClient, 15*time.Minute, 1)
	waitForReplicas(dummyDeploymentName, f.Namespace.ObjectMeta.Name, kubeClient, 15*time.Minute, 1)
}

func createDeploymentsToScale(f *framework.Framework, cs clientset.Interface) error {
	_, err := cs.Extensions().Deployments(f.Namespace.ObjectMeta.Name).Create(monitoring.StackdriverExporterDeployment(stackdriverExporterDeployment, f.Namespace.Name, 2, 100))
	if err != nil {
		return err
	}
	_, err = cs.CoreV1().Pods(f.Namespace.ObjectMeta.Name).Create(monitoring.StackdriverExporterPod(stackdriverExporterPod, f.Namespace.Name, stackdriverExporterPod, monitoring.CustomMetricName, 100))
	if err != nil {
		return err
	}
	_, err = cs.Extensions().Deployments(f.Namespace.ObjectMeta.Name).Create(monitoring.StackdriverExporterDeployment(dummyDeploymentName, f.Namespace.Name, 2, 100))
	return err
}

func cleanupDeploymentsToScale(f *framework.Framework, cs clientset.Interface) {
	_ = cs.Extensions().Deployments(f.Namespace.ObjectMeta.Name).Delete(stackdriverExporterDeployment, &metav1.DeleteOptions{})
	_ = cs.CoreV1().Pods(f.Namespace.ObjectMeta.Name).Delete(stackdriverExporterPod, &metav1.DeleteOptions{})
	_ = cs.Extensions().Deployments(f.Namespace.ObjectMeta.Name).Delete(dummyDeploymentName, &metav1.DeleteOptions{})
}

func createPodsHPA(f *framework.Framework, cs clientset.Interface) error {
	var minReplicas int32 = 1
	_, err := cs.AutoscalingV2beta1().HorizontalPodAutoscalers(f.Namespace.ObjectMeta.Name).Create(&as.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "custom-metrics-pods-hpa",
			Namespace: f.Namespace.ObjectMeta.Name,
		},
		Spec: as.HorizontalPodAutoscalerSpec{
			Metrics: []as.MetricSpec{
				{
					Type: as.PodsMetricSourceType,
					Pods: &as.PodsMetricSource{
						MetricName:         monitoring.CustomMetricName,
						TargetAverageValue: *resource.NewQuantity(200, resource.DecimalSI),
					},
				},
			},
			MaxReplicas: 3,
			MinReplicas: &minReplicas,
			ScaleTargetRef: as.CrossVersionObjectReference{
				APIVersion: "extensions/v1beta1",
				Kind:       "Deployment",
				Name:       stackdriverExporterDeployment,
			},
		},
	})
	return err
}

func createObjectHPA(f *framework.Framework, cs clientset.Interface) error {
	var minReplicas int32 = 1
	_, err := cs.AutoscalingV2beta1().HorizontalPodAutoscalers(f.Namespace.ObjectMeta.Name).Create(&as.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "custom-metrics-objects-hpa",
			Namespace: f.Namespace.ObjectMeta.Name,
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
						TargetValue: *resource.NewQuantity(200, resource.DecimalSI),
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
	})
	return err
}

func waitForReplicas(deploymentName, namespace string, cs clientset.Interface, timeout time.Duration, desiredReplicas int) {
	interval := 20 * time.Second
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		deployment, err := cs.Extensions().Deployments(namespace).Get(deploymentName, metav1.GetOptions{})
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
