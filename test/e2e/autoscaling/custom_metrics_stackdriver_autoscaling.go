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
	"fmt"
	"math"
	"strings"
	"time"

	gcm "google.golang.org/api/monitoring/v3"
	"google.golang.org/api/option"
	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/instrumentation/monitoring"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"golang.org/x/oauth2/google"
)

const (
	stackdriverExporterDeployment = "stackdriver-exporter-deployment"
	dummyDeploymentName           = "dummy-deployment"
	stackdriverExporterPod        = "stackdriver-exporter-pod"
	externalMetricValue           = int64(85)
)

type externalMetricTarget struct {
	value     int64
	isAverage bool
}

var _ = SIGDescribe("[HPA]", feature.CustomMetricsAutoscaling, "Horizontal pod autoscaling (scale resource: Custom Metrics from Stackdriver)", func() {
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("gce", "gke")
	})

	f := framework.NewDefaultFramework("horizontal-pod-autoscaling")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Describe("with Custom Metric of type Pod from Stackdriver", func() {
		ginkgo.It("should scale down", func(ctx context.Context) {
			initialReplicas := 2
			// metric should cause scale down
			metricValue := int64(100)
			metricTarget := 2 * metricValue
			metricSpecs := []autoscalingv2.MetricSpec{
				podMetricSpecWithAverageValueTarget(monitoring.CustomMetricName, metricTarget),
			}
			tc := CustomMetricTestCase{
				framework:       f,
				kubeClient:      f.ClientSet,
				initialReplicas: initialReplicas,
				scaledReplicas:  1,
				deployment:      monitoring.SimpleStackdriverExporterDeployment(stackdriverExporterDeployment, f.Namespace.ObjectMeta.Name, int32(initialReplicas), metricValue),
				hpa:             hpa("custom-metrics-pods-hpa", f.Namespace.ObjectMeta.Name, stackdriverExporterDeployment, 1, 3, metricSpecs),
			}
			tc.Run(ctx)
		})

		ginkgo.It("should scale up with two metrics", func(ctx context.Context) {
			initialReplicas := 1
			// metric 1 would cause a scale down, if not for metric 2
			metric1Value := int64(100)
			metric1Target := 2 * metric1Value
			// metric2 should cause a scale up
			metric2Value := int64(200)
			metric2Target := int64(0.5 * float64(metric2Value))
			metricSpecs := []autoscalingv2.MetricSpec{
				podMetricSpecWithAverageValueTarget("metric1", metric1Target),
				podMetricSpecWithAverageValueTarget("metric2", metric2Target),
			}
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
			tc := CustomMetricTestCase{
				framework:       f,
				kubeClient:      f.ClientSet,
				initialReplicas: initialReplicas,
				scaledReplicas:  3,
				deployment:      monitoring.StackdriverExporterDeployment(stackdriverExporterDeployment, f.Namespace.ObjectMeta.Name, int32(initialReplicas), containers),
				hpa:             hpa("custom-metrics-pods-hpa", f.Namespace.ObjectMeta.Name, stackdriverExporterDeployment, 1, 3, metricSpecs),
			}
			tc.Run(ctx)
		})

		ginkgo.It("should scale down with Prometheus", func(ctx context.Context) {
			initialReplicas := 2
			// metric should cause scale down
			metricValue := int64(100)
			metricTarget := 2 * metricValue
			metricSpecs := []autoscalingv2.MetricSpec{
				podMetricSpecWithAverageValueTarget(monitoring.CustomMetricName, metricTarget),
			}
			tc := CustomMetricTestCase{
				framework:       f,
				kubeClient:      f.ClientSet,
				initialReplicas: initialReplicas,
				scaledReplicas:  1,
				deployment:      monitoring.PrometheusExporterDeployment(stackdriverExporterDeployment, f.Namespace.ObjectMeta.Name, int32(initialReplicas), metricValue),
				hpa:             hpa("custom-metrics-pods-hpa", f.Namespace.ObjectMeta.Name, stackdriverExporterDeployment, 1, 3, metricSpecs),
			}
			tc.Run(ctx)
		})
	})

	ginkgo.Describe("with Custom Metric of type Object from Stackdriver", func() {
		ginkgo.It("should scale down", func(ctx context.Context) {
			initialReplicas := 2
			// metric should cause scale down
			metricValue := int64(100)
			metricTarget := 2 * metricValue
			metricSpecs := []autoscalingv2.MetricSpec{
				objectMetricSpecWithValueTarget(metricTarget),
			}
			tc := CustomMetricTestCase{
				framework:       f,
				kubeClient:      f.ClientSet,
				initialReplicas: initialReplicas,
				scaledReplicas:  1,
				deployment:      noExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas)),
				pod:             monitoring.StackdriverExporterPod(stackdriverExporterPod, f.Namespace.Name, stackdriverExporterPod, monitoring.CustomMetricName, metricValue),
				hpa:             hpa("custom-metrics-objects-hpa", f.Namespace.ObjectMeta.Name, dummyDeploymentName, 1, 3, metricSpecs),
			}
			tc.Run(ctx)
		})

		ginkgo.It("should scale down to 0", func(ctx context.Context) {
			initialReplicas := 2
			// metric should cause scale down
			metricValue := int64(0)
			metricTarget := int64(200)
			metricSpecs := []autoscalingv2.MetricSpec{
				objectMetricSpecWithValueTarget(metricTarget),
			}
			tc := CustomMetricTestCase{
				framework:       f,
				kubeClient:      f.ClientSet,
				initialReplicas: initialReplicas,
				scaledReplicas:  0,
				deployment:      noExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas)),
				pod:             monitoring.StackdriverExporterPod(stackdriverExporterPod, f.Namespace.Name, stackdriverExporterPod, monitoring.CustomMetricName, metricValue),
				hpa:             hpa("custom-metrics-objects-hpa", f.Namespace.ObjectMeta.Name, dummyDeploymentName, 0, 3, metricSpecs),
			}
			tc.Run(ctx)
		})
	})

	ginkgo.Describe("with External Metric from Stackdriver", func() {
		ginkgo.It("should scale down with target value", func(ctx context.Context) {
			initialReplicas := 2
			// metric should cause scale down
			metricValue := externalMetricValue
			metricTarget := 3 * metricValue
			metricSpecs := []autoscalingv2.MetricSpec{
				externalMetricSpecWithTarget("target", f.Namespace.ObjectMeta.Name, externalMetricTarget{
					value:     metricTarget,
					isAverage: false,
				}),
			}
			tc := CustomMetricTestCase{
				framework:       f,
				kubeClient:      f.ClientSet,
				initialReplicas: initialReplicas,
				scaledReplicas:  1,
				deployment:      noExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas)),
				pod:             monitoring.StackdriverExporterPod(stackdriverExporterPod, f.Namespace.Name, stackdriverExporterPod, "target", metricValue),
				hpa:             hpa("custom-metrics-external-hpa", f.Namespace.ObjectMeta.Name, dummyDeploymentName, 1, 3, metricSpecs),
			}
			tc.Run(ctx)
		})

		ginkgo.It("should scale down with target average value", func(ctx context.Context) {
			initialReplicas := 2
			// metric should cause scale down
			metricValue := externalMetricValue
			metricAverageTarget := 3 * metricValue
			metricSpecs := []autoscalingv2.MetricSpec{
				externalMetricSpecWithTarget("target_average", f.Namespace.ObjectMeta.Name, externalMetricTarget{
					value:     metricAverageTarget,
					isAverage: true,
				}),
			}
			tc := CustomMetricTestCase{
				framework:       f,
				kubeClient:      f.ClientSet,
				initialReplicas: initialReplicas,
				scaledReplicas:  1,
				deployment:      noExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas)),
				pod:             monitoring.StackdriverExporterPod(stackdriverExporterPod, f.Namespace.Name, stackdriverExporterPod, "target_average", externalMetricValue),
				hpa:             hpa("custom-metrics-external-hpa", f.Namespace.ObjectMeta.Name, dummyDeploymentName, 1, 3, metricSpecs),
			}
			tc.Run(ctx)
		})

		ginkgo.It("should scale up with two metrics", func(ctx context.Context) {
			initialReplicas := 1
			// metric 1 would cause a scale down, if not for metric 2
			metric1Value := externalMetricValue
			metric1Target := 2 * metric1Value
			// metric2 should cause a scale up
			metric2Value := externalMetricValue
			metric2Target := int64(math.Ceil(0.5 * float64(metric2Value)))
			metricSpecs := []autoscalingv2.MetricSpec{
				externalMetricSpecWithTarget("external_metric_1", f.Namespace.ObjectMeta.Name, externalMetricTarget{
					value:     metric1Target,
					isAverage: true,
				}),
				externalMetricSpecWithTarget("external_metric_2", f.Namespace.ObjectMeta.Name, externalMetricTarget{
					value:     metric2Target,
					isAverage: true,
				}),
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
				hpa:             hpa("custom-metrics-external-hpa", f.Namespace.ObjectMeta.Name, dummyDeploymentName, 1, 3, metricSpecs),
			}
			tc.Run(ctx)
		})
	})

	ginkgo.Describe("with multiple metrics of different types", func() {
		ginkgo.It("should scale up when one metric is missing (Pod and External metrics)", func(ctx context.Context) {
			initialReplicas := 1
			// First metric a pod metric which is missing.
			// Second metric is external metric which is present, it should cause scale up.
			metricSpecs := []autoscalingv2.MetricSpec{
				podMetricSpecWithAverageValueTarget(monitoring.CustomMetricName, 2*externalMetricValue),
				externalMetricSpecWithTarget("external_metric", f.Namespace.ObjectMeta.Name, externalMetricTarget{
					value:     int64(math.Ceil(0.5 * float64(externalMetricValue))),
					isAverage: true,
				}),
			}
			containers := []monitoring.CustomMetricContainerSpec{
				{
					Name:        "stackdriver-exporter-metric",
					MetricName:  "external_metric",
					MetricValue: externalMetricValue,
				},
				// Pod Resource metric is missing from here.
			}
			tc := CustomMetricTestCase{
				framework:       f,
				kubeClient:      f.ClientSet,
				initialReplicas: initialReplicas,
				scaledReplicas:  3,
				deployment:      monitoring.StackdriverExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas), containers),
				hpa:             hpa("multiple-metrics", f.Namespace.ObjectMeta.Name, dummyDeploymentName, 1, 3, metricSpecs)}
			tc.Run(ctx)
		})

		ginkgo.It("should scale up when one metric is missing (Resource and Object metrics)", func(ctx context.Context) {
			initialReplicas := 1
			metricValue := int64(100)
			// First metric a resource metric which is missing (no consumption).
			// Second metric is object metric which is present, it should cause scale up.
			metricSpecs := []autoscalingv2.MetricSpec{
				resourceMetricSpecWithAverageUtilizationTarget(50),
				objectMetricSpecWithValueTarget(int64(math.Ceil(0.5 * float64(metricValue)))),
			}
			tc := CustomMetricTestCase{
				framework:       f,
				kubeClient:      f.ClientSet,
				initialReplicas: initialReplicas,
				scaledReplicas:  3,
				deployment:      monitoring.SimpleStackdriverExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas), 0),
				pod:             monitoring.StackdriverExporterPod(stackdriverExporterPod, f.Namespace.Name, stackdriverExporterPod, monitoring.CustomMetricName, metricValue),
				hpa:             hpa("multiple-metrics", f.Namespace.ObjectMeta.Name, dummyDeploymentName, 1, 3, metricSpecs)}
			tc.Run(ctx)
		})

		ginkgo.It("should not scale down when one metric is missing (Container Resource and External Metrics)", func(ctx context.Context) {
			initialReplicas := 2
			// First metric a container resource metric which is missing.
			// Second metric is external metric which is present, it should cause scale down if the first metric wasn't missing.
			metricSpecs := []autoscalingv2.MetricSpec{
				containerResourceMetricSpecWithAverageUtilizationTarget("container-resource-metric", 50),
				externalMetricSpecWithTarget("external_metric", f.Namespace.ObjectMeta.Name, externalMetricTarget{
					value:     2 * externalMetricValue,
					isAverage: true,
				}),
			}
			containers := []monitoring.CustomMetricContainerSpec{
				{
					Name:        "stackdriver-exporter-metric",
					MetricName:  "external_metric",
					MetricValue: externalMetricValue,
				},
				// Container Resource metric is missing from here.
			}
			tc := CustomMetricTestCase{
				framework:       f,
				kubeClient:      f.ClientSet,
				initialReplicas: initialReplicas,
				scaledReplicas:  initialReplicas,
				verifyStability: true,
				deployment:      monitoring.StackdriverExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas), containers),
				hpa:             hpa("multiple-metrics", f.Namespace.ObjectMeta.Name, dummyDeploymentName, 1, 3, metricSpecs)}
			tc.Run(ctx)
		})

		ginkgo.It("should not scale down when one metric is missing (Pod and Object Metrics)", func(ctx context.Context) {
			initialReplicas := 2
			metricValue := int64(100)
			// First metric an object metric which is missing.
			// Second metric is pod metric which is present, it should cause scale down if the first metric wasn't missing.
			metricSpecs := []autoscalingv2.MetricSpec{
				objectMetricSpecWithValueTarget(int64(math.Ceil(0.5 * float64(metricValue)))),
				podMetricSpecWithAverageValueTarget("pod_metric", 2*metricValue),
			}
			containers := []monitoring.CustomMetricContainerSpec{
				{
					Name:        "stackdriver-exporter-metric",
					MetricName:  "pod_metric",
					MetricValue: metricValue,
				},
			}
			tc := CustomMetricTestCase{
				framework:       f,
				kubeClient:      f.ClientSet,
				initialReplicas: initialReplicas,
				scaledReplicas:  initialReplicas,
				verifyStability: true,
				deployment:      monitoring.StackdriverExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas), containers),
				hpa:             hpa("multiple-metrics", f.Namespace.ObjectMeta.Name, dummyDeploymentName, 1, 3, metricSpecs)}
			tc.Run(ctx)
		})
	})

})

// CustomMetricTestCase is a struct for test cases.
type CustomMetricTestCase struct {
	framework       *framework.Framework
	hpa             *autoscalingv2.HorizontalPodAutoscaler
	kubeClient      clientset.Interface
	deployment      *appsv1.Deployment
	pod             *v1.Pod
	initialReplicas int
	scaledReplicas  int
	verifyStability bool
}

// Run starts test case.
func (tc *CustomMetricTestCase) Run(ctx context.Context) {
	projectID := framework.TestContext.CloudConfig.ProjectID

	client, err := google.DefaultClient(ctx, gcm.CloudPlatformScope)
	if err != nil {
		framework.Failf("Failed to initialize gcm default client, %v", err)
	}

	// Hack for running tests locally, needed to authenticate in Stackdriver
	// If this is your use case, create application default credentials:
	// $ gcloud auth application-default login
	// and uncomment following lines:

	// ts, err := google.DefaultTokenSource(oauth2.NoContext)
	// framework.Logf("Couldn't get application default credentials, %v", err)
	// if err != nil {
	// 	framework.Failf("Error accessing application default credentials, %v", err)
	// }
	// client = oauth2.NewClient(oauth2.NoContext, ts)

	gcmService, err := gcm.NewService(ctx, option.WithHTTPClient(client))
	if err != nil {
		framework.Failf("Failed to create gcm service, %v", err)
	}

	// Set up a cluster: create a custom metric and set up k8s-sd adapter
	err = monitoring.CreateDescriptors(gcmService, projectID)
	if err != nil {
		if strings.Contains(err.Error(), "Request throttled") {
			e2eskipper.Skipf("Skipping...hitting rate limits on creating and updating metrics/labels")
		}
		framework.Failf("Failed to create metric descriptor: %v", err)
	}
	defer monitoring.CleanupDescriptors(gcmService, projectID)

	err = monitoring.CreateAdapter(monitoring.AdapterDefault)
	defer monitoring.CleanupAdapter(monitoring.AdapterDefault)
	if err != nil {
		framework.Failf("Failed to set up: %v", err)
	}

	// Run application that exports the metric
	err = createDeploymentToScale(ctx, tc.framework, tc.kubeClient, tc.deployment, tc.pod)
	if err != nil {
		framework.Failf("Failed to create stackdriver-exporter pod: %v", err)
	}
	ginkgo.DeferCleanup(cleanupDeploymentsToScale, tc.framework, tc.kubeClient, tc.deployment, tc.pod)

	// Wait for the deployment to run
	waitForReplicas(ctx, tc.deployment.ObjectMeta.Name, tc.framework.Namespace.ObjectMeta.Name, tc.kubeClient, 15*time.Minute, tc.initialReplicas)

	// Autoscale the deployment
	_, err = tc.kubeClient.AutoscalingV2().HorizontalPodAutoscalers(tc.framework.Namespace.ObjectMeta.Name).Create(ctx, tc.hpa, metav1.CreateOptions{})
	if err != nil {
		framework.Failf("Failed to create HPA: %v", err)
	}
	ginkgo.DeferCleanup(framework.IgnoreNotFound(tc.kubeClient.AutoscalingV2().HorizontalPodAutoscalers(tc.framework.Namespace.ObjectMeta.Name).Delete), tc.hpa.ObjectMeta.Name, metav1.DeleteOptions{})

	waitForReplicas(ctx, tc.deployment.ObjectMeta.Name, tc.framework.Namespace.ObjectMeta.Name, tc.kubeClient, 15*time.Minute, tc.scaledReplicas)

	if tc.verifyStability {
		ensureDesiredReplicasInRange(ctx, tc.deployment.ObjectMeta.Name, tc.framework.Namespace.ObjectMeta.Name, tc.kubeClient, tc.scaledReplicas, tc.scaledReplicas, 10*time.Minute)
	}
}

func createDeploymentToScale(ctx context.Context, f *framework.Framework, cs clientset.Interface, deployment *appsv1.Deployment, pod *v1.Pod) error {
	if deployment != nil {
		_, err := cs.AppsV1().Deployments(f.Namespace.ObjectMeta.Name).Create(ctx, deployment, metav1.CreateOptions{})
		if err != nil {
			return err
		}
	}
	if pod != nil {
		_, err := cs.CoreV1().Pods(f.Namespace.ObjectMeta.Name).Create(ctx, pod, metav1.CreateOptions{})
		if err != nil {
			return err
		}
	}
	return nil
}

func cleanupDeploymentsToScale(ctx context.Context, f *framework.Framework, cs clientset.Interface, deployment *appsv1.Deployment, pod *v1.Pod) {
	if deployment != nil {
		_ = cs.AppsV1().Deployments(f.Namespace.ObjectMeta.Name).Delete(ctx, deployment.ObjectMeta.Name, metav1.DeleteOptions{})
	}
	if pod != nil {
		_ = cs.CoreV1().Pods(f.Namespace.ObjectMeta.Name).Delete(ctx, pod.ObjectMeta.Name, metav1.DeleteOptions{})
	}
}

func podMetricSpecWithAverageValueTarget(metric string, targetValue int64) autoscalingv2.MetricSpec {
	return autoscalingv2.MetricSpec{
		Type: autoscalingv2.PodsMetricSourceType,
		Pods: &autoscalingv2.PodsMetricSource{
			Metric: autoscalingv2.MetricIdentifier{
				Name: metric,
			},
			Target: autoscalingv2.MetricTarget{
				Type:         autoscalingv2.AverageValueMetricType,
				AverageValue: resource.NewQuantity(targetValue, resource.DecimalSI),
			},
		},
	}
}

func objectMetricSpecWithValueTarget(targetValue int64) autoscalingv2.MetricSpec {
	return autoscalingv2.MetricSpec{
		Type: autoscalingv2.ObjectMetricSourceType,
		Object: &autoscalingv2.ObjectMetricSource{
			Metric: autoscalingv2.MetricIdentifier{
				Name: monitoring.CustomMetricName,
			},
			DescribedObject: autoscalingv2.CrossVersionObjectReference{
				Kind: "Pod",
				Name: stackdriverExporterPod,
			},
			Target: autoscalingv2.MetricTarget{
				Type:  autoscalingv2.ValueMetricType,
				Value: resource.NewQuantity(targetValue, resource.DecimalSI),
			},
		},
	}
}

func resourceMetricSpecWithAverageUtilizationTarget(targetValue int32) autoscalingv2.MetricSpec {
	return autoscalingv2.MetricSpec{
		Type: autoscalingv2.ResourceMetricSourceType,
		Resource: &autoscalingv2.ResourceMetricSource{
			Name: v1.ResourceCPU,
			Target: autoscalingv2.MetricTarget{
				Type:               autoscalingv2.UtilizationMetricType,
				AverageUtilization: &targetValue,
			},
		},
	}
}

func containerResourceMetricSpecWithAverageUtilizationTarget(containerName string, targetValue int32) autoscalingv2.MetricSpec {
	return autoscalingv2.MetricSpec{
		Type: autoscalingv2.ContainerResourceMetricSourceType,
		ContainerResource: &autoscalingv2.ContainerResourceMetricSource{
			Name:      v1.ResourceCPU,
			Container: containerName,
			Target: autoscalingv2.MetricTarget{
				Type:               autoscalingv2.UtilizationMetricType,
				AverageUtilization: &targetValue,
			},
		},
	}
}

func externalMetricSpecWithTarget(metric string, namespace string, target externalMetricTarget) autoscalingv2.MetricSpec {
	selector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"resource.type": "k8s_pod"},
		MatchExpressions: []metav1.LabelSelectorRequirement{
			{
				Key:      "resource.labels.namespace_name",
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{namespace},
			},
			{
				Key:      "resource.labels.pod_name",
				Operator: metav1.LabelSelectorOpExists,
				Values:   []string{},
			},
		},
	}
	metricSpec := autoscalingv2.MetricSpec{
		Type: autoscalingv2.ExternalMetricSourceType,
		External: &autoscalingv2.ExternalMetricSource{
			Metric: autoscalingv2.MetricIdentifier{
				Name:     "custom.googleapis.com|" + metric,
				Selector: selector,
			},
		},
	}
	if target.isAverage {
		metricSpec.External.Target.Type = autoscalingv2.AverageValueMetricType
		metricSpec.External.Target.AverageValue = resource.NewQuantity(target.value, resource.DecimalSI)
	} else {
		metricSpec.External.Target.Type = autoscalingv2.ValueMetricType
		metricSpec.External.Target.Value = resource.NewQuantity(target.value, resource.DecimalSI)
	}
	return metricSpec
}

func hpa(name, namespace, deploymentName string, minReplicas, maxReplicas int32, metricSpecs []autoscalingv2.MetricSpec) *autoscalingv2.HorizontalPodAutoscaler {
	return &autoscalingv2.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
			Metrics:     metricSpecs,
			MinReplicas: &minReplicas,
			MaxReplicas: maxReplicas,
			ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       deploymentName,
			},
		},
	}
}

func waitForReplicas(ctx context.Context, deploymentName, namespace string, cs clientset.Interface, timeout time.Duration, desiredReplicas int) {
	interval := 20 * time.Second
	err := wait.PollUntilContextTimeout(ctx, interval, timeout, true, func(ctx context.Context) (bool, error) {
		deployment, err := cs.AppsV1().Deployments(namespace).Get(ctx, deploymentName, metav1.GetOptions{})
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

func ensureDesiredReplicasInRange(ctx context.Context, deploymentName, namespace string, cs clientset.Interface, minDesiredReplicas, maxDesiredReplicas int, timeout time.Duration) {
	interval := 60 * time.Second
	err := wait.PollUntilContextTimeout(ctx, interval, timeout, true, func(ctx context.Context) (bool, error) {
		deployment, err := cs.AppsV1().Deployments(namespace).Get(ctx, deploymentName, metav1.GetOptions{})
		if err != nil {
			return true, err
		}
		replicas := int(deployment.Status.ReadyReplicas)
		framework.Logf("expecting there to be in [%d, %d] replicas (are: %d)", minDesiredReplicas, maxDesiredReplicas, replicas)
		if replicas < minDesiredReplicas {
			return false, fmt.Errorf("number of replicas below target")
		} else if replicas > maxDesiredReplicas {
			return false, fmt.Errorf("number of replicas above target")
		} else {
			return false, nil // Expected number of replicas found. Continue polling until timeout.
		}
	})
	// The call above always returns an error, but if it is timeout, it's OK (condition satisfied all the time).
	if wait.Interrupted(err) || strings.Contains(err.Error(), "would exceed context deadline") {
		framework.Logf("Number of replicas was stable over %v", timeout)
		return
	}
	framework.ExpectNoErrorWithOffset(1, err)
}

func noExporterDeployment(name, namespace string, replicas int32) *appsv1.Deployment {
	d := e2edeployment.NewDeployment(name, replicas, map[string]string{"name": name}, "", "", appsv1.RollingUpdateDeploymentStrategyType)
	d.ObjectMeta.Namespace = namespace
	d.Spec.Template.Spec = v1.PodSpec{Containers: []v1.Container{
		{
			Name:            "sleeper",
			Image:           "registry.k8s.io/e2e-test-images/agnhost:2.40",
			ImagePullPolicy: v1.PullAlways,
			Command:         []string{"/agnhost"},
			Args:            []string{"pause"}, // do nothing forever
		},
	}}
	return d
}
