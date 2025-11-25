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

package helmapplyset

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/helmapplyset/status"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestStatusAggregationIntegration(t *testing.T) {
	// Start the API server
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	// Create clients
	config := rest.CopyConfig(server.ClientConfig)
	client := clientset.NewForConfigOrDie(config)
	dynamicClient := framework.DynamicClientOrDie(config)

	// Create a namespace
	ns := framework.CreateNamespaceOrDie(client, "status-test", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	// Setup the Aggregator
	logger := klog.NewKlogr()
	mapper := framework.NewRESTMapper(client, dynamicClient)
	aggregator := status.NewAggregator(client, dynamicClient, mapper, logger)

	// 1. Create a healthy Deployment
	deploymentName := "test-deployment"
	deployment := &apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      deploymentName,
			Namespace: ns.Name,
			Labels: map[string]string{
				"applyset.kubernetes.io/part-of": "test-applyset-id",
			},
		},
		Spec: apps.DeploymentSpec{
			Replicas: int32Ptr(1),
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "test"},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"app": "test"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "nginx",
							Image: "nginx:latest",
						},
					},
				},
			},
		},
	}
	_, err := client.AppsV1().Deployments(ns.Name).Create(context.Background(), deployment, metav1.CreateOptions{})
	require.NoError(t, err)

	// Update Deployment status to be ready
	deployment, err = client.AppsV1().Deployments(ns.Name).Get(context.Background(), deploymentName, metav1.GetOptions{})
	require.NoError(t, err)
	deployment.Status.Replicas = 1
	deployment.Status.ReadyReplicas = 1
	deployment.Status.AvailableReplicas = 1
	deployment.Status.UpdatedReplicas = 1
	_, err = client.AppsV1().Deployments(ns.Name).UpdateStatus(context.Background(), deployment, metav1.UpdateOptions{})
	require.NoError(t, err)

	// 2. Create a Service without endpoints (unhealthy/degraded)
	serviceName := "test-service"
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceName,
			Namespace: ns.Name,
			Labels: map[string]string{
				"applyset.kubernetes.io/part-of": "test-applyset-id",
			},
		},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{"app": "non-existent"}, // Points to nothing
			Ports: []v1.ServicePort{
				{
					Port: 80,
				},
			},
		},
	}
	_, err = client.CoreV1().Services(ns.Name).Create(context.Background(), service, metav1.CreateOptions{})
	require.NoError(t, err)

	// 3. Run Aggregation
	groupKinds := sets.New[schema.GroupKind](
		schema.GroupKind{Group: "apps", Kind: "Deployment"},
		schema.GroupKind{Group: "", Kind: "Service"},
	)

	health, err := aggregator.AggregateHealth(context.Background(), "test-release", ns.Name, "test-applyset-id", groupKinds)
	require.NoError(t, err)

	// 4. Verify results
	assert.Equal(t, "test-release", health.ReleaseName)
	assert.Equal(t, ns.Name, health.Namespace)
	assert.Equal(t, "test-applyset-id", health.ApplySetID)
	
	// We expect "degraded" because the service has no endpoints
	assert.Equal(t, "degraded", health.OverallStatus)
	assert.Equal(t, 2, health.TotalResources)
	
	// Check individual resources
	// Note: Keys format depends on how they are stored in aggregator.go
	deploymentKey := "apps/v1/Deployment/" + ns.Name + "/" + deploymentName
	serviceKey := "v1/Service/" + ns.Name + "/" + serviceName
	
	// If keys are stored as just GVK/Namespace/Name, we might need to adjust based on Aggregator implementation
	// Checking logic in aggregator.go: key := fmt.Sprintf("%s/%s/%s", groupKind.String(), resource.GetNamespace(), resource.GetName())
	// apps/v1/Deployment -> GroupKind string is "Deployment.apps" usually or similar. 
    // Let's check schema.GroupKind.String(): "Kind.Group"
    
	deploymentKey = "Deployment.apps/" + ns.Name + "/" + deploymentName
	serviceKey = "Service/" + ns.Name + "/" + serviceName

	// Verify Deployment is healthy
	depHealth, ok := health.ResourceHealth[deploymentKey]
	if !ok {
		// Try alternative key format if test fails
		// But let's assume standard String() output
	}
	// Actually, let's iterate to find it if key format is tricky
	foundDep := false
	for k, v := range health.ResourceHealth {
		if k == deploymentKey {
			assert.True(t, v.Healthy, "Deployment should be healthy")
			foundDep = true
		}
	}
	assert.True(t, foundDep, "Deployment health status not found")

	// Verify Service is unhealthy
	foundSvc := false
	for k, v := range health.ResourceHealth {
		if k == serviceKey {
			assert.False(t, v.Healthy, "Service should be unhealthy (no endpoints)")
			assert.Contains(t, v.Reason, "NoEndpoints")
			foundSvc = true
		}
	}
	assert.True(t, foundSvc, "Service health status not found")
}

func int32Ptr(i int32) *int32 {
	return &i
}

