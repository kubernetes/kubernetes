/*
Copyright The Kubernetes Authors.

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

package conditionalauthorization

import (
	"context"
	"fmt"
	"testing"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/utils/ptr"
)

func hpaTestCases() []conditionalAuthzTestCase {
	return []conditionalAuthzTestCase{
		// Tests for HPA v1 and v2 with CPU utilization conditions.
		// The authorizer returns version-specific CEL conditions that require
		// the target CPU utilization to be at most 80%.
		{
			name:        "hpa v1 cpu utilization - allow",
			user:        "hpa-cpu-allow-user",
			authorizers: celConditionalAuthorizerVariants(hpaCPUUtilizationDecision),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.AutoscalingV1().HorizontalPodAutoscalers("test-ns").Create(context.TODO(), &autoscalingv1.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "hpa-v1-allow" + suffix},
					Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
							Kind:       "Deployment",
							Name:       "test-deploy",
							APIVersion: "apps/v1",
						},
						MaxReplicas:                    10,
						TargetCPUUtilizationPercentage: ptr.To[int32](80),
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: new(false),
		},
		{
			name:        "hpa v1 cpu utilization - deny",
			user:        "hpa-cpu-deny-user",
			authorizers: celConditionalAuthorizerVariants(hpaCPUUtilizationDecision),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.AutoscalingV1().HorizontalPodAutoscalers("test-ns").Create(context.TODO(), &autoscalingv1.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "hpa-v1-deny" + suffix},
					Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
							Kind:       "Deployment",
							Name:       "test-deploy",
							APIVersion: "apps/v1",
						},
						MaxReplicas:                    10,
						TargetCPUUtilizationPercentage: ptr.To[int32](90),
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name:        "hpa v2 cpu utilization - allow",
			user:        "hpa-cpu-allow-user",
			authorizers: celConditionalAuthorizerVariants(hpaCPUUtilizationDecision),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.AutoscalingV2().HorizontalPodAutoscalers("test-ns").Create(context.TODO(), &autoscalingv2.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "hpa-v2-allow" + suffix},
					Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
							Kind:       "Deployment",
							Name:       "test-deploy",
							APIVersion: "apps/v1",
						},
						MaxReplicas: 10,
						Metrics: []autoscalingv2.MetricSpec{
							{
								Type: autoscalingv2.ResourceMetricSourceType,
								Resource: &autoscalingv2.ResourceMetricSource{
									Name: corev1.ResourceCPU,
									Target: autoscalingv2.MetricTarget{
										Type:               autoscalingv2.UtilizationMetricType,
										AverageUtilization: ptr.To[int32](80),
									},
								},
							},
						},
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: new(false),
		},
		{
			name:        "hpa v2 cpu utilization - deny",
			user:        "hpa-cpu-deny-user",
			authorizers: celConditionalAuthorizerVariants(hpaCPUUtilizationDecision),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				_, err := client.AutoscalingV2().HorizontalPodAutoscalers("test-ns").Create(context.TODO(), &autoscalingv2.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "hpa-v2-deny" + suffix},
					Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
							Kind:       "Deployment",
							Name:       "test-deploy",
							APIVersion: "apps/v1",
						},
						MaxReplicas: 10,
						Metrics: []autoscalingv2.MetricSpec{
							{
								Type: autoscalingv2.ResourceMetricSourceType,
								Resource: &autoscalingv2.ResourceMetricSource{
									Name: corev1.ResourceCPU,
									Target: autoscalingv2.MetricTarget{
										Type:               autoscalingv2.UtilizationMetricType,
										AverageUtilization: ptr.To[int32](90),
									},
								},
							},
						},
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name:        "hpa v1 cpu utilization - update allowed",
			user:        "hpa-cpu-allow-update-user",
			authorizers: celConditionalAuthorizerVariants(hpaCPUUtilizationDecision),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				// Create with 70% (allowed: 70 <= 80)
				created, err := client.AutoscalingV1().HorizontalPodAutoscalers("test-ns").Create(context.TODO(), &autoscalingv1.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "hpa-v1-update-allow" + suffix},
					Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
							Kind:       "Deployment",
							Name:       "test-deploy",
							APIVersion: "apps/v1",
						},
						MaxReplicas:                    10,
						TargetCPUUtilizationPercentage: ptr.To[int32](70),
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("initial create should have succeeded: %w", err)
				}
				// Update to 80% (allowed: new=80<=80 && old=70<=80)
				created.Spec.TargetCPUUtilizationPercentage = ptr.To[int32](80)
				_, err = client.AutoscalingV1().HorizontalPodAutoscalers("test-ns").Update(context.TODO(), created, metav1.UpdateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: new(false),
		},
		{
			name:        "hpa v1 cpu utilization - update denied",
			user:        "hpa-cpu-deny-update-user",
			authorizers: celConditionalAuthorizerVariants(hpaCPUUtilizationDecision),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				// Create with 70% (allowed: 70 <= 80)
				created, err := client.AutoscalingV1().HorizontalPodAutoscalers("test-ns").Create(context.TODO(), &autoscalingv1.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "hpa-v1-update-deny" + suffix},
					Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
							Kind:       "Deployment",
							Name:       "test-deploy",
							APIVersion: "apps/v1",
						},
						MaxReplicas:                    10,
						TargetCPUUtilizationPercentage: ptr.To[int32](70),
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("initial create should have succeeded: %w", err)
				}
				// Update to 90% (denied: new=90>80)
				created.Spec.TargetCPUUtilizationPercentage = ptr.To[int32](90)
				_, err = client.AutoscalingV1().HorizontalPodAutoscalers("test-ns").Update(context.TODO(), created, metav1.UpdateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name:        "hpa v2 cpu utilization - update allowed",
			user:        "hpa-cpu-allow-update-v2-user",
			authorizers: celConditionalAuthorizerVariants(hpaCPUUtilizationDecision),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				// Create with 70% (allowed: 70 <= 80)
				created, err := client.AutoscalingV2().HorizontalPodAutoscalers("test-ns").Create(context.TODO(), &autoscalingv2.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "hpa-v2-update-allow" + suffix},
					Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
							Kind:       "Deployment",
							Name:       "test-deploy",
							APIVersion: "apps/v1",
						},
						MaxReplicas: 10,
						Metrics: []autoscalingv2.MetricSpec{
							{
								Type: autoscalingv2.ResourceMetricSourceType,
								Resource: &autoscalingv2.ResourceMetricSource{
									Name: corev1.ResourceCPU,
									Target: autoscalingv2.MetricTarget{
										Type:               autoscalingv2.UtilizationMetricType,
										AverageUtilization: ptr.To[int32](70),
									},
								},
							},
						},
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("initial create should have succeeded: %w", err)
				}
				// Update to 80% (allowed: new=80<=80 && old=70<=80)
				created.Spec.Metrics[0].Resource.Target.AverageUtilization = ptr.To[int32](80)
				_, err = client.AutoscalingV2().HorizontalPodAutoscalers("test-ns").Update(context.TODO(), created, metav1.UpdateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: new(false),
		},
		{
			name:        "hpa v2 cpu utilization - update denied",
			user:        "hpa-cpu-deny-update-v2-user",
			authorizers: celConditionalAuthorizerVariants(hpaCPUUtilizationDecision),
			makeRequest: func(t *testing.T, client *clientset.Clientset, suffix string) error {
				// Create with 70% (allowed: 70 <= 80)
				created, err := client.AutoscalingV2().HorizontalPodAutoscalers("test-ns").Create(context.TODO(), &autoscalingv2.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "hpa-v2-update-deny" + suffix},
					Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
							Kind:       "Deployment",
							Name:       "test-deploy",
							APIVersion: "apps/v1",
						},
						MaxReplicas: 10,
						Metrics: []autoscalingv2.MetricSpec{
							{
								Type: autoscalingv2.ResourceMetricSourceType,
								Resource: &autoscalingv2.ResourceMetricSource{
									Name: corev1.ResourceCPU,
									Target: autoscalingv2.MetricTarget{
										Type:               autoscalingv2.UtilizationMetricType,
										AverageUtilization: ptr.To[int32](70),
									},
								},
							},
						},
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("initial create should have succeeded: %w", err)
				}
				// Update to 90% (denied: new=90>80)
				created.Spec.Metrics[0].Resource.Target.AverageUtilization = ptr.To[int32](90)
				_, err = client.AutoscalingV2().HorizontalPodAutoscalers("test-ns").Update(context.TODO(), created, metav1.UpdateOptions{})
				return err
			},
			expectAllowed: false,
		},
	}
}

// hpaCPUUtilizationDecision is a decisionFunc for HPA CPU-utilization-based
// tests. It emits a ConditionsMap that only allows HPAs whose CPU utilization
// is at most 80%. The CEL expression is version-specific: for v1 it checks
// spec.targetCPUUtilizationPercentage, for v2 it iterates spec.metrics to
// find the CPU resource metric and checks averageUtilization. For updates,
// both the old and new objects must satisfy the condition.
func hpaCPUUtilizationDecision(a authorizer.Attributes, conditionsType string) authorizer.ConditionsAwareDecision {
	if a.GetResource() != "horizontalpodautoscalers" {
		return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
	}

	var objectCondition, oldObjectCondition string
	switch a.GetAPIVersion() {
	case "v1":
		objectCondition = `has(object.spec.targetCPUUtilizationPercentage) && object.spec.targetCPUUtilizationPercentage <= 80`
		oldObjectCondition = `has(oldObject.spec.targetCPUUtilizationPercentage) && oldObject.spec.targetCPUUtilizationPercentage <= 80`
	default: // v2, v2beta2, etc.
		objectCondition = `has(object.spec.metrics) && object.spec.metrics.exists(m, ` +
			`m.type == "Resource" && ` +
			`has(m.resource) && ` +
			`m.resource.name == "cpu" && ` +
			`has(m.resource.target) && ` +
			`m.resource.target.type == "Utilization" && ` +
			`has(m.resource.target.averageUtilization) && ` +
			`m.resource.target.averageUtilization <= 80)`
		oldObjectCondition = `has(oldObject.spec.metrics) && oldObject.spec.metrics.exists(m, ` +
			`m.type == "Resource" && ` +
			`has(m.resource) && ` +
			`m.resource.name == "cpu" && ` +
			`has(m.resource.target) && ` +
			`m.resource.target.type == "Utilization" && ` +
			`has(m.resource.target.averageUtilization) && ` +
			`m.resource.target.averageUtilization <= 80)`
	}

	var condition string
	switch a.GetVerb() {
	case "create":
		condition = objectCondition
	case "update":
		condition = objectCondition + " && " + oldObjectCondition
	default:
		return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
	}

	return authorizer.ConditionsAwareDecisionConditionsMap(
		nil, nil,
		[]authorizer.Condition{
			authorizer.GenericCondition{
				ID:          "example.com/limit-cpu-utilization",
				Condition:   condition,
				Type:        conditionsType,
				Description: "only allow HPAs with CPU utilization at most 80%",
			},
		},
	)
}
