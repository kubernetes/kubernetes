/*
Copyright 2016 The Kubernetes Authors.

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

package cmd

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestKubeletMaintenanceAddedRemoved(t *testing.T) {
	podList := getPods()
	node := getNode()
	f, tf, codec, ns := NewAPIFactory()
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch {
			case req.Method == "PUT" && strings.HasPrefix(req.URL.Path, "/api/v1/namespaces/test/pods"):
				updatePod := api.Pod{}
				if err := decodeUpdate(req, &updatePod, codec); err != nil {
					t.Fatalf("Unexpected error %v", err)
				}
				for i, pod := range podList.Items {
					if pod.Name == updatePod.Name {
						podList.Items[i] = updatePod
					}
				}
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &updatePod)}, nil
			case req.Method == "PUT" && strings.HasPrefix(req.URL.Path, "/api/v1/nodes"):
				updateNode := &api.Node{}
				if err := decodeUpdate(req, updateNode, codec); err != nil {
					t.Fatalf("Unexpected error %v", err)
				}
				node = updateNode
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, node)}, nil
			case req.URL.Path == "/api/v1/pods":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, podList)}, nil
			case req.URL.Path == "/api/v1/nodes/node":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, node)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.ClientConfig = defaultClientConfig()
	tf.Namespace = "test"

	t.Logf("Adding taints and tolerations")
	out := bytes.NewBuffer([]byte{})
	cmd := NewCmdMaintenance(f, out)
	cmd.Run(cmd, []string{node.Name, "on"})
	taints, _ := api.GetTaintsFromNodeAnnotations(node.Annotations)
	if len(taints) != 1 {
		t.Fatalf("Expected 1 taint on %v", node.Annotations)
	}
	for _, taint := range taints {
		if taint.Key != unversioned.TaintNodeMaintenance {
			t.Fatalf("Expected %v to be the only key in %v", unversioned.TaintNodeMaintenance, node.Annotations)
		}
	}
	for _, pod := range podList.Items {
		tolerations, _ := api.GetTolerationsFromPodAnnotations(pod.Annotations)
		if len(tolerations) != 1 {
			t.Fatalf("Expected 1 toleration on %v", pod.Annotations)
		}
		for _, toleration := range tolerations {
			if toleration.Key != unversioned.TaintNodeOutage {
				t.Fatalf("Expected %v to be the only key in %v", unversioned.TaintNodeOutage, node.Annotations)
			}
		}
	}

	t.Logf("Removing added taints and tolerations")
	out = bytes.NewBuffer([]byte{})
	cmd = NewCmdMaintenance(f, out)
	cmd.Run(cmd, []string{node.Name, "off"})
	taints, _ = api.GetTaintsFromNodeAnnotations(node.Annotations)
	if len(taints) != 0 {
		t.Fatalf("Expected 0 taints on %v", node.Annotations)
	}
	for _, pod := range podList.Items {
		tolerations, _ := api.GetTolerationsFromPodAnnotations(pod.Annotations)
		if len(tolerations) != 0 {
			t.Fatalf("Expected 0 tolerations on %v", pod.Annotations)
		}
	}
}

func decodeUpdate(req *http.Request, into runtime.Object, codec runtime.Codec) error {
	data, err := ioutil.ReadAll(req.Body)
	if err != nil {
		return err
	}
	defer req.Body.Close()
	if err := runtime.DecodeInto(codec, data, into); err != nil {
		return err
	}
	return nil
}

func getNode() *api.Node {
	return &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name:              "node",
			CreationTimestamp: unversioned.Time{Time: time.Now()},
		},
		Spec: api.NodeSpec{
			ExternalID: "node",
		},
		Status: api.NodeStatus{},
	}
}

func getPods() *api.PodList {
	return &api.PodList{
		ListMeta: unversioned.ListMeta{
			ResourceVersion: "15",
		},
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "bar", Namespace: "test", ResourceVersion: "11"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
		},
	}
}
