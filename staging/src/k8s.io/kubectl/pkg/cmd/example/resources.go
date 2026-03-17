/*
Copyright 2025 The Kubernetes Authors.

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

package example

import (
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

const (
	podDefaultImage        = "alpine:latest"
	deploymentDefaultImage = "nginx:stable"
)

func buildPod(name, image string) *corev1.Pod {
	podImage := image
	if podImage == "" {
		podImage = podDefaultImage
	}

	return &corev1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"app.kubernetes.io/name": name,
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:    name,
					Image:   podImage,
					Command: []string{"sleep", "3600"},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("250m"),
							corev1.ResourceMemory: resource.MustParse("64Mi"),
						},
						Limits: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("500m"),
							corev1.ResourceMemory: resource.MustParse("128Mi"),
						},
					},
				},
			},
		},
	}
}

func buildDeployment(name, image string, replicas int) *appsv1.Deployment {
	deploymentImage := image
	if deploymentImage == "" {
		deploymentImage = deploymentDefaultImage
	}

	replicaCount := int32(1)
	if replicas > 0 {
		replicaCount = int32(replicas)
	}

	labels := map[string]string{"app.kubernetes.io/name": name}
	selector := &metav1.LabelSelector{MatchLabels: labels}

	return &appsv1.Deployment{
		TypeMeta: metav1.TypeMeta{APIVersion: appsv1.SchemeGroupVersion.String(), Kind: "Deployment"},
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicaCount,
			Selector: selector,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  name,
							Image: deploymentImage,
							Ports: []corev1.ContainerPort{{ContainerPort: 80}},
						},
					},
				},
			},
		},
	}
}

func buildService(name string) *corev1.Service {
	return &corev1.Service{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Service"},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: corev1.ServiceSpec{
			Type: corev1.ServiceTypeClusterIP,
			Selector: map[string]string{
				"app.kubernetes.io/name": name,
			},
			Ports: []corev1.ServicePort{
				{
					Port:       80,
					TargetPort: intstr.FromInt32(80),
				},
			},
		},
	}
}

func buildPVC(name string) *corev1.PersistentVolumeClaim {
	return &corev1.PersistentVolumeClaim{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "PersistentVolumeClaim"},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
			Resources: corev1.VolumeResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceStorage: resource.MustParse("1Gi"),
				},
			},
		},
	}
}

func buildSecret(name string) *corev1.Secret {
	return &corev1.Secret{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Secret"},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Type: corev1.SecretTypeOpaque,
		StringData: map[string]string{
			"username": "user",
			"password": "pass",
		},
	}
}

func buildCRD(_ string) map[string]interface{} {
	return map[string]interface{}{
		"apiVersion": "apiextensions.k8s.io/v1",
		"kind":       "CustomResourceDefinition",
		"metadata": map[string]interface{}{
			"name": "widgets.example.com",
		},
		"spec": map[string]interface{}{
			"group": "example.com",
			"versions": []interface{}{
				map[string]interface{}{
					"name":    "v1",
					"served":  true,
					"storage": true,
					"schema": map[string]interface{}{
						"openAPIV3Schema": map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"spec": map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"size": map[string]interface{}{
											"type": "string",
										},
									},
								},
							},
						},
					},
				},
			},
			"scope": "Namespaced",
			"names": map[string]interface{}{
				"plural":     "widgets",
				"singular":   "widget",
				"kind":       "Widget",
				"shortNames": []interface{}{"w"},
			},
		},
	}
}

func buildConfigMap(name string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "ConfigMap"},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"app.kubernetes.io/name": name,
			},
		},
		Data: map[string]string{
			"config.yaml": "key: value\n",
			"LOG_LEVEL":   "info",
		},
	}
}

func buildJob(name, image string) *batchv1.Job {
	jobImage := image
	if jobImage == "" {
		jobImage = "perl:5.40"
	}

	return &batchv1.Job{
		TypeMeta: metav1.TypeMeta{APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"app.kubernetes.io/name": name,
			},
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:    name,
							Image:   jobImage,
							Command: []string{"perl", "-Mbignum=bpi", "-wle", "print bpi(2000)"},
						},
					},
					RestartPolicy: corev1.RestartPolicyNever,
				},
			},
			BackoffLimit: int32Ptr(4),
		},
	}
}

func buildCronJob(name, image string) *batchv1.CronJob {
	cronImage := image
	if cronImage == "" {
		cronImage = "busybox:1.36"
	}

	return &batchv1.CronJob{
		TypeMeta: metav1.TypeMeta{APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "CronJob"},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"app.kubernetes.io/name": name,
			},
		},
		Spec: batchv1.CronJobSpec{
			Schedule: "*/5 * * * *",
			JobTemplate: batchv1.JobTemplateSpec{
				Spec: batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:    name,
									Image:   cronImage,
									Command: []string{"/bin/sh", "-c", "date; echo Hello from the Kubernetes cluster"},
								},
							},
							RestartPolicy: corev1.RestartPolicyOnFailure,
						},
					},
				},
			},
		},
	}
}

func buildIngress(name string) *networkingv1.Ingress {
	pathType := networkingv1.PathTypePrefix
	return &networkingv1.Ingress{
		TypeMeta: metav1.TypeMeta{APIVersion: networkingv1.SchemeGroupVersion.String(), Kind: "Ingress"},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"app.kubernetes.io/name": name,
			},
			Annotations: map[string]string{
				"nginx.ingress.kubernetes.io/rewrite-target": "/",
			},
		},
		Spec: networkingv1.IngressSpec{
			Rules: []networkingv1.IngressRule{
				{
					Host: "example.com",
					IngressRuleValue: networkingv1.IngressRuleValue{
						HTTP: &networkingv1.HTTPIngressRuleValue{
							Paths: []networkingv1.HTTPIngressPath{
								{
									Path:     "/",
									PathType: &pathType,
									Backend: networkingv1.IngressBackend{
										Service: &networkingv1.IngressServiceBackend{
											Name: name,
											Port: networkingv1.ServiceBackendPort{
												Number: 80,
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
}

func buildNetworkPolicy(name string) *networkingv1.NetworkPolicy {
	return &networkingv1.NetworkPolicy{
		TypeMeta: metav1.TypeMeta{APIVersion: networkingv1.SchemeGroupVersion.String(), Kind: "NetworkPolicy"},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"app.kubernetes.io/name": name,
			},
		},
		Spec: networkingv1.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app.kubernetes.io/name": name,
				},
			},
			PolicyTypes: []networkingv1.PolicyType{
				networkingv1.PolicyTypeIngress,
				networkingv1.PolicyTypeEgress,
			},
			Ingress: []networkingv1.NetworkPolicyIngressRule{
				{
					From: []networkingv1.NetworkPolicyPeer{
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"app.kubernetes.io/name": "frontend",
								},
							},
						},
					},
					Ports: []networkingv1.NetworkPolicyPort{
						{
							Port: intstrPtr(intstr.FromInt32(80)),
						},
					},
				},
			},
			Egress: []networkingv1.NetworkPolicyEgressRule{
				{
					To: []networkingv1.NetworkPolicyPeer{
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"app.kubernetes.io/name": "database",
								},
							},
						},
					},
					Ports: []networkingv1.NetworkPolicyPort{
						{
							Port: intstrPtr(intstr.FromInt32(5432)),
						},
					},
				},
			},
		},
	}
}

// buildGateway returns a Gateway API Gateway resource using an unstructured map
// because sigs.k8s.io/gateway-api types are not vendored in kubectl.
// This demonstrates the Gateway API approach to traffic ingress — the modern,
// role-oriented successor to Ingress resources.
func buildGateway(name string) map[string]interface{} {
	return map[string]interface{}{
		"apiVersion": "gateway.networking.k8s.io/v1",
		"kind":       "Gateway",
		"metadata": map[string]interface{}{
			"name": name,
			"labels": map[string]interface{}{
				"app.kubernetes.io/name": name,
			},
		},
		"spec": map[string]interface{}{
			"gatewayClassName": "example",
			"listeners": []interface{}{
				map[string]interface{}{
					"name":     "http",
					"protocol": "HTTP",
					"port":     int64(80),
					"allowedRoutes": map[string]interface{}{
						"namespaces": map[string]interface{}{
							"from": "Same",
						},
					},
				},
				map[string]interface{}{
					"name":     "https",
					"protocol": "HTTPS",
					"port":     int64(443),
					"tls": map[string]interface{}{
						"mode": "Terminate",
						"certificateRefs": []interface{}{
							map[string]interface{}{
								"name": "example-cert",
							},
						},
					},
					"allowedRoutes": map[string]interface{}{
						"namespaces": map[string]interface{}{
							"from": "Same",
						},
					},
				},
			},
		},
	}
}

// buildHTTPRoute returns a Gateway API HTTPRoute resource using an unstructured map
// because sigs.k8s.io/gateway-api types are not vendored in kubectl.
// This demonstrates how HTTPRoute attaches to a Gateway to define routing rules —
// a cleaner separation of concerns compared to the monolithic Ingress resource.
func buildHTTPRoute(name string) map[string]interface{} {
	return map[string]interface{}{
		"apiVersion": "gateway.networking.k8s.io/v1",
		"kind":       "HTTPRoute",
		"metadata": map[string]interface{}{
			"name": name,
			"labels": map[string]interface{}{
				"app.kubernetes.io/name": name,
			},
		},
		"spec": map[string]interface{}{
			"parentRefs": []interface{}{
				map[string]interface{}{
					"name": "example-gateway",
				},
			},
			"hostnames": []interface{}{
				"example.com",
			},
			"rules": []interface{}{
				map[string]interface{}{
					"matches": []interface{}{
						map[string]interface{}{
							"path": map[string]interface{}{
								"type":  "PathPrefix",
								"value": "/api",
							},
						},
					},
					"backendRefs": []interface{}{
						map[string]interface{}{
							"name": "api-service",
							"port": int64(80),
						},
					},
				},
				map[string]interface{}{
					"matches": []interface{}{
						map[string]interface{}{
							"path": map[string]interface{}{
								"type":  "PathPrefix",
								"value": "/",
							},
						},
					},
					"backendRefs": []interface{}{
						map[string]interface{}{
							"name": "frontend-service",
							"port": int64(80),
						},
					},
				},
			},
		},
	}
}

func int32Ptr(i int32) *int32 {
	return &i
}

func intstrPtr(v intstr.IntOrString) *intstr.IntOrString {
	return &v
}
