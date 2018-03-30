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

package kubeletclient

import (
	"fmt"
	"io/ioutil"
	"net/http"

	"k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/kubelet/client"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// KubeletClient represents a Kublet Client
type KubeletClient struct {
	config    client.KubeletClientConfig
	transport http.RoundTripper
}

// CreateClient creates a KubeletClient
func CreateClient() (KubeletClient, error) {
	kubeletClient := KubeletClient{}
	kubeletClient.config = client.KubeletClientConfig{
		Port:        10250,
		EnableHttps: true,
		TLSClientConfig: restclient.TLSClientConfig{
			CertFile: "/etc/kubernetes/pki/apiserver-kubelet-client.crt",
			KeyFile:  "/etc/kubernetes/pki/apiserver-kubelet-client.key",
			Insecure: true,
		},
	}
	rt, err := client.MakeTransport(&kubeletClient.config)
	if err != nil {
		return kubeletClient, err
	}
	kubeletClient.transport = rt
	return kubeletClient, nil
}

// GetStaticPodConfigHash returns the config hash for a named static pod
func (k KubeletClient) GetStaticPodConfigHash(name string) (string, error) {
	pod, err := k.GetPod(name)
	if err != nil {
		return "", err
	}
	annotations := pod.ObjectMeta.GetAnnotations()
	return annotations[kubetypes.ConfigHashAnnotationKey], nil
}

// GetPod returns the pod matching name
func (k KubeletClient) GetPod(name string) (*core.Pod, error) {
	podList, err := k.GetPods()
	if err != nil {
		return nil, err
	}
	for _, pod := range podList.Items {
		if pod.Name == name {
			return &pod, nil
		}
	}
	return nil, fmt.Errorf("Pod not found")
}

// GetPods returns the current PodList from the Kubelet
func (k KubeletClient) GetPods() (*core.PodList, error) {
	req, _ := http.NewRequest("GET", "https://localhost:10250/pods", nil)
	resp, err := k.transport.RoundTrip(req)
	if err != nil {
		return nil, fmt.Errorf("kubelet request failed: %v", err)
	}
	codec := legacyscheme.Codecs.LegacyCodec(schema.GroupVersion{Group: v1.GroupName, Version: "v1"})
	body, _ := ioutil.ReadAll(resp.Body)
	obj, _ := runtime.Decode(codec, body)
	pods := obj.(*core.PodList)
	return pods, nil
}
