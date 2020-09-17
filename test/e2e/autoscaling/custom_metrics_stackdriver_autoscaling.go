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
	"math"
	"time"

	gcm "google.golang.org/api/monitoring/v3"
	appsv1 "k8s.io/api/apps/v1"
	as "k8s.io/api/autoscaling/v2beta1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/instrumentation/monitoring"

	"github.com/onsi/ginkgo"
	"golang.org/x/oauth2/google"
)

const (
	stackdriverExporterDeployment = "stackdriver-exporter-deployment"
	dummyDeploymentName           = "dummy-deployment"
	stackdriverExporterPod        = "stackdriver-exporter-pod"
	externalMetricValue           = int64(85)
)

var _ = SIGDescribe("[HPA] Horizontal pod autoscaling (scale resource: Custom Metrics from Stackdriver)", func() {
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("gce", "gke")
	})

	f := framework.NewDefaultFramework("horizontal-pod-autoscaling")

	ginkgo.It("should scale down with Custom Metric of type Pod from Stackdriver [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 2
		// metric should cause scale down
		metricValue := int64(100)
		metricTarget := 2 * metricValue
		tc := CustomMetricTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialReplicas: initialReplicas,
			scaledReplicas:  1,
			deployment:      monitoring.SimpleStackdriverExporterDeployment(stackdriverExporterDeployment, f.Namespace.ObjectMeta.Name, int32(initialReplicas), metricValue),
			hpa:             simplePodsHPA(f.Namespace.ObjectMeta.Name, metricTarget)}
		tc.Run()
	})

	ginkgo.It("should scale down with Custom Metric of type Object from Stackdriver [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 2
		// metric should cause scale down
		metricValue := int64(100)
		metricTarget := 2 * metricValue
		tc := CustomMetricTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialReplicas: initialReplicas,
			scaledReplicas:  1,
			// Metric exported by deployment is ignored
			deployment: monitoring.SimpleStackdriverExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas), 0 /* ignored */),
			pod:        monitoring.StackdriverExporterPod(stackdriverExporterPod, f.Namespace.Name, stackdriverExporterPod, monitoring.CustomMetricName, metricValue),
			hpa:        objectHPA(f.Namespace.ObjectMeta.Name, metricTarget)}
		tc.Run()
	})

	ginkgo.It("should scale down with External Metric with target value from Stackdriver [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 2
		// metric should cause scale down
		metricValue := externalMetricValue
		metricTarget := 3 * metricValue
		metricTargets := map[string]externalMetricTarget{
			"target": {
				value:     metricTarget,
				isAverage: false,
			},
		}
		tc := CustomMetricTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialReplicas: initialReplicas,
			scaledReplicas:  1,
			// Metric exported by deployment is ignored
			deployment: monitoring.SimpleStackdriverExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas), 0 /* ignored */),
			pod:        monitoring.StackdriverExporterPod(stackdriverExporterPod, f.Namespace.Name, stackdriverExporterPod, "target", metricValue),
			hpa:        externalHPA(f.Namespace.ObjectMeta.Name, metricTargets)}
		tc.Run()
	})

	ginkgo.It("should scale down with External Metric with target average value from Stackdriver [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 2
		// metric should cause scale down
		metricValue := externalMetricValue
		metricAverageTarget := 3 * metricValue
		metricTargets := map[string]externalMetricTarget{
			"target_average": {
				value:     metricAverageTarget,
				isAverage: true,
			},
		}
		tc := CustomMetricTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialReplicas: initialReplicas,
			scaledReplicas:  1,
			// Metric exported by deployment is ignored
			deployment: monitoring.SimpleStackdriverExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas), 0 /* ignored */),
			pod:        monitoring.StackdriverExporterPod(stackdriverExporterPod, f.Namespace.Name, stackdriverExporterPod, "target_average", externalMetricValue),
			hpa:        externalHPA(f.Namespace.ObjectMeta.Name, metricTargets)}
		tc.Run()
	})

	ginkgo.It("should scale down with Custom Metric of type Pod from Stackdriver with Prometheus [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 2
		// metric should cause scale down
		metricValue := int64(100)
		metricTarget := 2 * metricValue
		tc := CustomMetricTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialReplicas: initialReplicas,
			scaledReplicas:  1,
			deployment:      monitoring.PrometheusExporterDeployment(stackdriverExporterDeployment, f.Namespace.ObjectMeta.Name, int32(initialReplicas), metricValue),
			hpa:             simplePodsHPA(f.Namespace.ObjectMeta.Name, metricTarget)}
		tc.Run()
	})

	ginkgo.It("should scale up with two metrics of type Pod from Stackdriver [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 1
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
		tc := CustomMetricTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialReplicas: initialReplicas,
			scaledReplicas:  3,
			deployment:      monitoring.StackdriverExporterDeployment(stackdriverExporterDeployment, f.Namespace.ObjectMeta.Name, int32(initialReplicas), containers),
			hpa:             podsHPA(f.Namespace.ObjectMeta.Name, stackdriverExporterDeployment, metricTargets)}
		tc.Run()
	})

	ginkgo.It("should scale up with two External metrics from Stackdriver [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 1
		// metric 1 would cause a scale down, if not for metric 2
		metric1Value := externalMetricValue
		metric1Target := 2 * metric1Value
		// metric2 should cause a scale up
		metric2Value := externalMetricValue
		metric2Target := int64(math.Ceil(0.5 * float64(metric2Value)))
		metricTargets := map[string]externalMetricTarget{
			"external_metric_1": {
				value:     metric1Target,
				isAverage: false,
			},
			"external_metric_2": {
				value:     metric2Target,
				isAverage: false,
			},
		}
		containers := []monitoring.CustomMetricContainerSpec{
			{
				Name:        "stackdriver-exporter-metric1",
				MetricName:  "external_metric_1",
				MetricValue: metric1Value,
			},
			{
				Name:        "stackdriver-exporter-metric2",
				MetricName:  "external_metric_2",
				MetricValue: metric2Value,
			},
		}
		tc := CustomMetricTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialReplicas: initialReplicas,
			scaledReplicas:  3,
			deployment:      monitoring.StackdriverExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas), containers),
			hpa:             externalHPA(f.Namespace.ObjectMeta.Name, metricTargets)}
		tc.Run()
	})
})

// CustomMetricTestCase is a struct for test cases.
type CustomMetricTestCase struct {
	framework       *framework.Framework
	hpa             *as.HorizontalPodAutoscaler
	kubeClient      clientset.Interface
	deployment      *appsv1.Deployment
	pod             *v1.Pod
	initialReplicas int
	scaledReplicas  int
}

// Run starts test case.
func (tc *CustomMetricTestCase) Run() {
	projectID := framework.TestContext.CloudConfig.ProjectID

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
	err = monitoring.CreateDescriptors(gcmService, projectID)
	if err != nil {
		framework.Failf("Failed to create metric descriptor: %v", err)
	}
	defer monitoring.CleanupDescriptors(gcmService, projectID)

	err = monitoring.CreateAdapter(monitoring.AdapterDefault)
	if err != nil {
		framework.Failf("Failed to set up: %v", err)
	}
	defer monitoring.CleanupAdapter(monitoring.AdapterDefault)

	// Run application that exports the metric
	err = createDeploymentToScale(tc.framework, tc.kubeClient, tc.deployment, tc.pod)
	if err != nil {
		framework.Failf("Failed to create stackdriver-exporter pod: %v", err)
	}
	defer cleanupDeploymentsToScale(tc.framework, tc.kubeClient, tc.deployment, tc.pod)

	// Wait for the deployment to run
	waitForReplicas(tc.deployment.ObjectMeta.Name, tc.framework.Namespace.ObjectMeta.Name, tc.kubeClient, 15*time.Minute, tc.initialReplicas)

	// Autoscale the deployment
	_, err = tc.kubeClient.AutoscalingV2beta1().HorizontalPodAutoscalers(tc.framework.Namespace.ObjectMeta.Name).Create(context.TODO(), tc.hpa, metav1.CreateOptions{})
	if err != nil {
		framework.Failf("Failed to create HPA: %v", err)
	}
	defer tc.kubeClient.AutoscalingV2beta1().HorizontalPodAutoscalers(tc.framework.Namespace.ObjectMeta.Name).Delete(context.TODO(), tc.hpa.ObjectMeta.Name, metav1.DeleteOptions{})

	waitForReplicas(tc.deployment.ObjectMeta.Name, tc.framework.Namespace.ObjectMeta.Name, tc.kubeClient, 15*time.Minute, tc.scaledReplicas)
}

func createDeploymentToScale(f *framework.Framework, cs clientset.Interface, deployment *appsv1.Deployment, pod *v1.Pod) error {
	if deployment != nil {
		_, err := cs.AppsV1().Deployments(f.Namespace.ObjectMeta.Name).Create(context.TODO(), deployment, metav1.CreateOptions{})
		if err != nil {
			return err
		}
	}
	if pod != nil {
		_, err := cs.CoreV1().Pods(f.Namespace.ObjectMeta.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		if err != nil {
			return err
		}
	}
	return nil
}

func cleanupDeploymentsToScale(f *framework.Framework, cs clientset.Interface, deployment *appsv1.Deployment, pod *v1.Pod) {
	if deployment != nil {
		_ = cs.AppsV1().Deployments(f.Namespace.ObjectMeta.Name).Delete(context.TODO(), deployment.ObjectMeta.Name, metav1.DeleteOptions{})
	}
	if pod != nil {
		_ = cs.CoreV1().Pods(f.Namespace.ObjectMeta.Name).Delete(context.TODO(), pod.ObjectMeta.Name, metav1.DeleteOptions{})
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
				APIVersion: "apps/v1",
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
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       dummyDeploymentName,
			},
		},
	}
}

type externalMetricTarget struct {
	value     int64
	isAverage bool
}

func externalHPA(namespace string, metricTargets map[string]externalMetricTarget) *as.HorizontalPodAutoscaler {
	var minReplicas int32 = 1
	metricSpecs := []as.MetricSpec{}
	selector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"resource.type": "gke_container"},
		MatchExpressions: []metav1.LabelSelectorRequirement{
			{
				Key:      "resource.labels.namespace_id",
				Operator: metav1.LabelSelectorOpIn,
				// TODO(bskiba): change default to real namespace name once it is available
				// from Stackdriver.
				Values: []string{"default", "dummy"},
			},
			{
				Key:      "resource.labels.pod_id",
				Operator: metav1.LabelSelectorOpExists,
				Values:   []string{},
			},
		},
	}
	for metric, target := range metricTargets {
		var metricSpec as.MetricSpec
		metricSpec = as.MetricSpec{
			Type: as.ExternalMetricSourceType,
			External: &as.ExternalMetricSource{
				MetricName:     "custom.googleapis.com|" + metric,
				MetricSelector: selector,
			},
		}
		if target.isAverage {
			metricSpec.External.TargetAverageValue = resource.NewQuantity(target.value, resource.DecimalSI)
		} else {
			metricSpec.External.TargetValue = resource.NewQuantity(target.value, resource.DecimalSI)
		}
		metricSpecs = append(metricSpecs, metricSpec)
	}
	hpa := &as.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "custom-metrics-external-hpa",
			Namespace: namespace,
		},
		Spec: as.HorizontalPodAutoscalerSpec{
			Metrics:     metricSpecs,
			MaxReplicas: 3,
			MinReplicas: &minReplicas,
			ScaleTargetRef: as.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       dummyDeploymentName,
			},
		},
	}

	return hpa
}

func waitForReplicas(deploymentName, namespace string, cs clientset.Interface, timeout time.Duration, desiredReplicas int) {
	interval := 20 * time.Second
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		deployment, err := cs.AppsV1().Deployments(namespace).Get(context.TODO(), deploymentName, metav1.GetOptions{})
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
