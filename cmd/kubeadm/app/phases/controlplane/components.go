/*
Copyright 2023 The Kubernetes Authors.

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

package controlplane

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/pkg/errors"
	"gopkg.in/yaml.v2"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
	kubeletconfig "k8s.io/kubelet/config/v1beta1"

	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

var ControlPlaneComponents = []string{
	kubeadmconstants.KubeAPIServer,
	kubeadmconstants.KubeControllerManager,
	kubeadmconstants.KubeScheduler,
}

type component struct {
	name    string
	labels  map[string]string
	touched bool
}

// WaitForControlPlaneComponent wait for control plane component to be ready by check pod status returned by kubelet
func WaitForControlPlaneComponents(componentNames []string, timeout time.Duration, manifestDir, kubeletDir, certificatesDir string) error {
	certFile := filepath.Join(certificatesDir, kubeadmconstants.APIServerKubeletClientCertName)
	keyFile := filepath.Join(certificatesDir, kubeadmconstants.APIServerKubeletClientKeyName)

	client, err := rest.HTTPClientFor(&rest.Config{
		TLSClientConfig: rest.TLSClientConfig{
			CertFile: certFile,
			KeyFile:  keyFile,
			Insecure: true,
		},
	})
	if err != nil {
		return errors.Wrap(err, "failed to create kubelet client")
	}

	kubeletEndpoint, err := getKubeletEndpoint(filepath.Join(kubeletDir, kubeadmconstants.KubeletConfigurationFileName))
	if err != nil {
		return errors.Wrap(err, "failed to get kubelet endpoint")
	}

	components := make([]*component, len(componentNames))
	for i, name := range componentNames {
		labels, err := getComponentLabels(name, manifestDir)
		if err != nil {
			return errors.Wrapf(err, "failed to get pod labels of %s component", name)
		}

		components[i] = &component{name, labels, false}
	}

	return wait.PollUntilContextTimeout(context.Background(), 5*time.Second, timeout, false, func(ctx context.Context) (bool, error) {
		klog.V(1).Infoln("[control-plane] polling status of control plane components...")

		resp, err := client.Get(kubeletEndpoint)
		if err != nil {
			fmt.Printf("[kubelet client] Error getting pods [%v]\n", err)
			return false, nil
		}

		defer resp.Body.Close()

		data, err := io.ReadAll(resp.Body)
		if err != nil {
			fmt.Printf("[kubelet client] Error reading pods from response body [%v]\n", err)
			return false, nil
		}

		pods := &v1.PodList{}
		if err := json.Unmarshal(data, pods); err != nil {
			fmt.Printf("[kubelet client] Error parsing pods from response body [%v]\n", err)
			return false, nil
		}

		for _, comp := range components {
			labels := comp.labels
		match_pod:
			for _, pod := range pods.Items {
				podLabels := pod.ObjectMeta.Labels
				for key, value := range labels {
					if podLabels[key] != value {
						continue match_pod
					}
				}

				comp.touched = true

				for _, status := range pod.Status.ContainerStatuses {
					if !status.Ready {
						klog.V(1).Infof("[control-plane] component: %s is not ready\n", comp.name)
						return false, nil
					}
				}

				klog.V(1).Infof("[control-plane] component: %s is ready\n", comp.name)
			}
		}

		for _, comp := range components {
			if !comp.touched {
				fmt.Printf("[kubelet client] Couldn`t find pod for component: %s with labels: [%v]\n", comp.name, comp.labels)
				return false, nil
			}
		}

		return true, nil
	})
}

func getKubeletEndpoint(configFile string) (string, error) {
	config := &kubeletconfig.KubeletConfiguration{}

	data, err := os.ReadFile(configFile)
	if err != nil {
		return "", err
	}

	if err := yaml.Unmarshal(data, config); err != nil {
		return "", err
	}

	kubeletPort := config.Port
	if kubeletPort == 0 {
		kubeletPort = kubeadmconstants.KubeletPort
	}

	return fmt.Sprintf("https://127.0.0.1:%d/pods", kubeletPort), nil
}

func getComponentLabels(component string, manifestDir string) (map[string]string, error) {
	pod, err := staticpod.ReadStaticPodFromDisk(kubeadmconstants.GetStaticPodFilepath(component, manifestDir))
	if err != nil {
		return nil, err
	}

	labels := pod.ObjectMeta.Labels
	if labels == nil {
		return nil, errors.New("Empty labels")
	}

	return labels, nil
}
