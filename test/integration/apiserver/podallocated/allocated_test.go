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

package podallocated

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kastesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestPodAllocatedSubresourceIntegration(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodAllocatedSubresource, true)
	tCtx := ktesting.Init(t)

	server := kastesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)
	client := clientset.NewForConfigOrDie(server.ClientConfig)

	const (
		nsName          = "allocated-test"
		podName         = "test-pod"
		allocatedCPUReq = "100m"
	)

	framework.CreateNamespaceOrDie(client, nsName, t)

	fakeAllocatedHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if requestedUID, found := strings.CutPrefix(r.URL.Path, "/allocatedPods/"); found {
			pod, err := client.CoreV1().Pods(nsName).Get(r.Context(), podName, metav1.GetOptions{})
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			if pod.UID != types.UID(requestedUID) {
				http.Error(w, fmt.Sprintf("UID mismatch! requested %s; found %s", requestedUID, pod.UID), http.StatusBadRequest)
				return
			}
			pod.Spec.Resources.Requests[v1.ResourceCPU] = resource.MustParse(allocatedCPUReq)
			pod.Status = v1.PodStatus{}

			if err := json.NewEncoder(w).Encode(pod); err != nil {
				panic(err) // Can't call require.NoError from within an http.Handler
			}
			return
		}
		w.WriteHeader(http.StatusNotFound)
	})
	node, err := startFakeNodeServer(t, client, fakeAllocatedHandler)
	require.NoError(t, err)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: nsName,
		},
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("200m"),
				},
			},
			NodeName: node.Name,
			Containers: []v1.Container{
				{Name: "c1", Image: "img1"},
			},
		},
	}
	pod, err = client.CoreV1().Pods(nsName).Create(tCtx, pod, metav1.CreateOptions{})
	require.NoError(t, err)

	// Query `/allocated` subresource
	result := client.CoreV1().RESTClient().Get().
		Namespace(nsName).
		Resource("pods").
		Name(podName).
		SubResource("allocated").
		Do(tCtx)

	require.NoError(t, result.Error())

	statusCode := 0
	result.StatusCode(&statusCode)
	require.Equal(t, http.StatusOK, statusCode)

	var allocatedPod v1.Pod
	err = result.Into(&allocatedPod)
	require.NoError(t, err, "failed to decode response into pod")

	assert.Equal(t, podName, allocatedPod.Name)
	assert.Equal(t, nsName, allocatedPod.Namespace)
	assert.Equal(t, pod.UID, allocatedPod.UID)
	assert.Equal(t, allocatedCPUReq, allocatedPod.Spec.Resources.Requests.Cpu().String())
	assert.Len(t, allocatedPod.Spec.Containers, 1)
	assert.Equal(t, "c1", allocatedPod.Spec.Containers[0].Name)
	assert.Equal(t, "img1", allocatedPod.Spec.Containers[0].Image)
	assert.Empty(t, allocatedPod.Status)
}

func startFakeNodeServer(tb ktesting.TB, cs clientset.Interface, handler http.Handler) (*v1.Node, error) {
	kubeletServer := httptest.NewTLSServer(handler)
	tb.Cleanup(kubeletServer.Close)

	kubeletURL, _ := url.Parse(kubeletServer.URL)
	kubeletHost := kubeletURL.Hostname()
	kubeletPort, _ := strconv.Atoi(kubeletURL.Port())

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{GenerateName: "fake-kubelet-"},
		Status: v1.NodeStatus{
			DaemonEndpoints: v1.NodeDaemonEndpoints{KubeletEndpoint: v1.DaemonEndpoint{Port: int32(kubeletPort)}},
			Addresses:       []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: kubeletHost}},
		},
	}
	return cs.CoreV1().Nodes().Create(context.Background(), node, metav1.CreateOptions{})
}
