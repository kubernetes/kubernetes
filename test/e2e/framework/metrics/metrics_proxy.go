/*
Copyright 2021 The Kubernetes Authors.

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

package metrics

import (
	"context"
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

type componentInfo struct {
	Name string
	Port int
	IP   string
}

// SetupMetricsProxy creates a nginx Pod to expose metrics from the secure port of kube-scheduler and kube-controller-manager in tests.
func SetupMetricsProxy(c clientset.Interface) error {
	var infos []componentInfo
	// The component pods might take some time to show up.
	err := wait.PollImmediate(time.Second*5, time.Minute*5, func() (bool, error) {
		podList, err := c.CoreV1().Pods(metav1.NamespaceSystem).List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return false, fmt.Errorf("list pods in ns %s: %w", metav1.NamespaceSystem, err)
		}
		var foundComponents []componentInfo
		for _, pod := range podList.Items {
			switch {
			case strings.HasPrefix(pod.Name, "kube-scheduler-"):
				foundComponents = append(foundComponents, componentInfo{
					Name: pod.Name,
					Port: kubeSchedulerPort,
					IP:   pod.Status.PodIP,
				})
			case strings.HasPrefix(pod.Name, "kube-controller-manager-"):
				foundComponents = append(foundComponents, componentInfo{
					Name: pod.Name,
					Port: kubeControllerManagerPort,
					IP:   pod.Status.PodIP,
				})
			}
		}
		if len(foundComponents) != 2 {
			klog.Infof("Only %d components found. Will retry.", len(foundComponents))
			klog.Infof("Found components: %v", foundComponents)
			return false, nil
		}
		infos = foundComponents
		return true, nil
	})
	if err != nil {
		return fmt.Errorf("missing component pods: %w", err)
	}

	klog.Infof("Found components: %v", infos)

	const name = metricsProxyPod
	_, err = c.CoreV1().ServiceAccounts(metav1.NamespaceSystem).Create(context.TODO(), &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{Name: name},
	}, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("create serviceAccount: %w", err)
	}
	_, err = c.RbacV1().ClusterRoles().Create(context.TODO(), &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Rules: []rbacv1.PolicyRule{
			{
				NonResourceURLs: []string{"/metrics"},
				Verbs:           []string{"get"},
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("create clusterRole: %w", err)
	}
	_, err = c.RbacV1().ClusterRoleBindings().Create(context.TODO(), &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Subjects: []rbacv1.Subject{
			{
				Kind:      rbacv1.ServiceAccountKind,
				Name:      name,
				Namespace: metav1.NamespaceSystem,
			},
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     name,
		},
	}, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("create clusterRoleBinding: %w", err)
	}

	var token string
	err = wait.PollImmediate(time.Second*5, time.Minute*5, func() (done bool, err error) {
		sa, err := c.CoreV1().ServiceAccounts(metav1.NamespaceSystem).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			klog.Warningf("Fail to get serviceAccount %s: %v", name, err)
			return false, nil
		}
		if len(sa.Secrets) < 1 {
			klog.Warningf("No secret found in serviceAccount %s", name)
			return false, nil
		}
		secretRef := sa.Secrets[0]
		secret, err := c.CoreV1().Secrets(metav1.NamespaceSystem).Get(context.TODO(), secretRef.Name, metav1.GetOptions{})
		if err != nil {
			klog.Warningf("Fail to get secret %s", secretRef.Name)
			return false, nil
		}
		token = string(secret.Data["token"])
		if len(token) == 0 {
			klog.Warningf("Token in secret %s is empty", secretRef.Name)
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		return err
	}

	var nginxConfig string
	for _, info := range infos {
		nginxConfig += fmt.Sprintf(`
server {
	listen %d;
	server_name _;
	proxy_set_header Authorization "Bearer %s";
	proxy_ssl_verify off;
	location /metrics {
		proxy_pass https://%s:%d;
	}
}
`, info.Port, token, info.IP, info.Port)
	}
	_, err = c.CoreV1().ConfigMaps(metav1.NamespaceSystem).Create(context.TODO(), &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			"metrics.conf": nginxConfig,
		},
	}, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("create nginx configmap: %w", err)
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceSystem,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:  "nginx",
				Image: imageutils.GetE2EImage(imageutils.Nginx),
				VolumeMounts: []v1.VolumeMount{{
					Name:      "config",
					MountPath: "/etc/nginx/conf.d",
					ReadOnly:  true,
				}},
			}},
			Volumes: []v1.Volume{{
				Name: "config",
				VolumeSource: v1.VolumeSource{
					ConfigMap: &v1.ConfigMapVolumeSource{
						LocalObjectReference: v1.LocalObjectReference{
							Name: name,
						},
					},
				},
			}},
		},
	}
	_, err = c.CoreV1().Pods(metav1.NamespaceSystem).Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		return err
	}
	err = e2epod.WaitForPodNameRunningInNamespace(c, name, metav1.NamespaceSystem)
	if err != nil {
		return err
	}
	klog.Info("Successfully setup metrics-proxy")
	return nil
}
