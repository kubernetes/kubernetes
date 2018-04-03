/*
Copyright 2018 The Kubernetes Authors.

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
	"time"

	appsv1 "k8s.io/api/apps/v1"
	autoscaling "k8s.io/api/autoscaling/v2beta1"
	corev1 "k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/instrumentation/monitoring"
)

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

func createStatefulSetsToScale(f *framework.Framework, cs clientset.Interface, deployment *appsv1.StatefulSet, pod *corev1.Pod) error {
	if deployment != nil {
		_, err := cs.Apps().StatefulSets(f.Namespace.ObjectMeta.Name).Create(deployment)
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

func cleanupStatefulSetsToScale(f *framework.Framework, cs clientset.Interface, deployment *appsv1.StatefulSet, pod *corev1.Pod) {
	if deployment != nil {
		_ = cs.Apps().StatefulSets(f.Namespace.ObjectMeta.Name).Delete(deployment.ObjectMeta.Name, &metav1.DeleteOptions{})
	}
	if pod != nil {
		_ = cs.CoreV1().Pods(f.Namespace.ObjectMeta.Name).Delete(pod.ObjectMeta.Name, &metav1.DeleteOptions{})
	}
}

func simplePodsHPA(namespace, targetRefName string, targetRefGroupVersionKind schema.GroupVersionKind, metricTarget int64) *autoscaling.HorizontalPodAutoscaler {
	return podsHPA(namespace, targetRefName, targetRefGroupVersionKind, map[string]int64{monitoring.CustomMetricName: metricTarget})
}

func podsHPA(namespace string, targetRefName string, targetGroupVersionKind schema.GroupVersionKind, metricTargets map[string]int64) *autoscaling.HorizontalPodAutoscaler {
	var minReplicas int32 = 1
	metrics := []autoscaling.MetricSpec{}
	for metric, target := range metricTargets {
		metrics = append(metrics, autoscaling.MetricSpec{
			Type: autoscaling.PodsMetricSourceType,
			Pods: &autoscaling.PodsMetricSource{
				MetricName:         metric,
				TargetAverageValue: *resource.NewQuantity(target, resource.DecimalSI),
			},
		})
	}
	return &autoscaling.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "custom-metrics-pods-hpa",
			Namespace: namespace,
		},
		Spec: autoscaling.HorizontalPodAutoscalerSpec{
			Metrics:     metrics,
			MaxReplicas: 3,
			MinReplicas: &minReplicas,
			ScaleTargetRef: autoscaling.CrossVersionObjectReference{
				APIVersion: targetGroupVersionKind.GroupVersion().String(),
				Kind:       targetGroupVersionKind.Kind,
				Name:       targetRefName,
			},
		},
	}
}

func objectHPA(namespace, podName string, targetRefName string, targetGroupVersionKind schema.GroupVersionKind, metricGroupKind schema.GroupKind, metricTarget int64) *autoscaling.HorizontalPodAutoscaler {
	var minReplicas int32 = 1
	return &autoscaling.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "custom-metrics-objects-hpa",
			Namespace: namespace,
		},
		Spec: autoscaling.HorizontalPodAutoscalerSpec{
			Metrics: []autoscaling.MetricSpec{
				{
					Type: autoscaling.ObjectMetricSourceType,
					Object: &autoscaling.ObjectMetricSource{
						MetricName: monitoring.CustomMetricName,
						Target: autoscaling.CrossVersionObjectReference{
							Kind: metricGroupKind.String(),
							Name: podName,
						},
						TargetValue: *resource.NewQuantity(metricTarget, resource.DecimalSI),
					},
				},
			},
			MaxReplicas: 3,
			MinReplicas: &minReplicas,
			ScaleTargetRef: autoscaling.CrossVersionObjectReference{
				APIVersion: targetGroupVersionKind.GroupVersion().String(),
				Kind:       targetGroupVersionKind.Kind,
				Name:       targetRefName,
			},
		},
	}
}

type externalMetricTarget struct {
	value     int64
	isAverage bool
}

func externalHPA(namespace string, metricTargets map[string]externalMetricTarget) *autoscaling.HorizontalPodAutoscaler {
	var minReplicas int32 = 1
	metricSpecs := []autoscaling.MetricSpec{}
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
		var metricSpec autoscaling.MetricSpec
		metricSpec = autoscaling.MetricSpec{
			Type: autoscaling.ExternalMetricSourceType,
			External: &autoscaling.ExternalMetricSource{
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
	hpa := &autoscaling.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "custom-metrics-external-hpa",
			Namespace: namespace,
		},
		Spec: autoscaling.HorizontalPodAutoscalerSpec{
			Metrics:     metricSpecs,
			MaxReplicas: 3,
			MinReplicas: &minReplicas,
			ScaleTargetRef: autoscaling.CrossVersionObjectReference{
				APIVersion: "extensions/v1beta1",
				Kind:       "Deployment",
				Name:       dummyDeploymentName,
			},
		},
	}

	return hpa
}

// waitForReplicas waits for the total number of pods selected by listOptions in the namespace to match desiredReplicas
func waitForReplicas(namespace string, cs clientset.Interface, timeout time.Duration, desiredReplicas int, listOptions metav1.ListOptions) {
	interval := 20 * time.Second
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		pods, err := cs.Core().Pods(namespace).List(listOptions)
		if err != nil {
			framework.Failf("Failed to get pod count for %s: %v", namespace, err)
		}
		replicas := len(pods.Items)
		framework.Logf("waiting for %d replicas (current: %d)", desiredReplicas, replicas)
		return replicas == desiredReplicas, nil // Expected number of replicas found. Exit.
	})
	if err != nil {
		framework.Failf("Timeout waiting %v for %v replicas", timeout, desiredReplicas)
	}
}
