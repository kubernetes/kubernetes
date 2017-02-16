// +build integration,!no-etcd

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

package ttlcontroller

import (
	"fmt"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	listers "k8s.io/kubernetes/pkg/client/listers/core/v1"
	"k8s.io/kubernetes/pkg/controller/ttl"
	"k8s.io/kubernetes/test/integration/framework"
)

func createClientAndInformers(t *testing.T, server *httptest.Server) (*clientset.Clientset, informers.SharedInformerFactory) {
	config := restclient.Config{
		Host:  server.URL,
		QPS:   500,
		Burst: 500,
	}
	testClient := clientset.NewForConfigOrDie(&config)

	informers := informers.NewSharedInformerFactory(testClient, time.Second)
	return testClient, informers
}

func createNodes(t *testing.T, client *clientset.Clientset, startIndex, endIndex int) {
	var wg sync.WaitGroup
	for i := startIndex; i < endIndex; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			node := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: fmt.Sprintf("node-%d", idx),
				},
			}
			if _, err := client.Core().Nodes().Create(node); err != nil {
				t.Fatalf("Failed to create node: %v", err)
			}
		}(i)
	}
	wg.Wait()
}

func deleteNodes(t *testing.T, client *clientset.Clientset, startIndex, endIndex int) {
	var wg sync.WaitGroup
	for i := startIndex; i < endIndex; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			name := fmt.Sprintf("node-%d", idx)
			if err := client.Core().Nodes().Delete(name, &metav1.DeleteOptions{}); err != nil {
				t.Fatalf("Failed to create node: %v", err)
			}
		}(i)
	}
	wg.Wait()
}

func waitForNodesWithTTLAnnotation(t *testing.T, nodeLister listers.NodeLister, numNodes, ttlSeconds int) {
	if err := wait.Poll(time.Second, 30*time.Second, func() (bool, error) {
		nodes, err := nodeLister.List(labels.Everything())
		if err != nil || len(nodes) != numNodes {
			return false, nil
		}
		for _, node := range nodes {
			if node.Annotations == nil {
				return false, nil
			}
			value, ok := node.Annotations[v1.ObjectTTLAnnotationKey]
			if !ok {
				return false, nil
			}
			currentTTL, err := strconv.Atoi(value)
			if err != nil || currentTTL != ttlSeconds {
				return false, nil
			}
		}
		return true, nil
	}); err != nil {
		t.Fatalf("Failed waiting for all nodes with annotation: %v", err)
	}
}

// Test whether ttlcontroller sets correct ttl annotations.
func TestTTLAnnotations(t *testing.T) {
	_, server := framework.RunAMaster(nil)
	defer server.Close()

	testClient, informers := createClientAndInformers(t, server)
	nodeInformer := informers.Core().V1().Nodes()
	ttlc := ttl.NewTTLController(nodeInformer, testClient)

	stopCh := make(chan struct{})
	defer close(stopCh)
	go nodeInformer.Informer().Run(stopCh)
	go ttlc.Run(1, stopCh)

	// Create 100 nodes all should have annotation equal to 0.
	createNodes(t, testClient, 0, 100)
	waitForNodesWithTTLAnnotation(t, informers.Core().V1().Nodes().Lister(), 100, 0)

	// Create 1 more node, all annotation should change to 15.
	createNodes(t, testClient, 100, 101)
	waitForNodesWithTTLAnnotation(t, informers.Core().V1().Nodes().Lister(), 101, 15)

	// Delete 11 nodes, it should still remain at the level of 15.
	deleteNodes(t, testClient, 90, 101)
	waitForNodesWithTTLAnnotation(t, informers.Core().V1().Nodes().Lister(), 90, 15)

	// Delete 1 more node, all should be decreased to 0.
	deleteNodes(t, testClient, 89, 90)
	waitForNodesWithTTLAnnotation(t, informers.Core().V1().Nodes().Lister(), 89, 0)
}
