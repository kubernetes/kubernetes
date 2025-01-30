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
	"context"
	"fmt"
	"strconv"
	"sync"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	listers "k8s.io/client-go/listers/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/klog/v2/ktesting"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/ttl"
	"k8s.io/kubernetes/test/integration/framework"
)

func createClientAndInformers(t *testing.T, server *kubeapiservertesting.TestServer) (*clientset.Clientset, informers.SharedInformerFactory) {
	config := restclient.CopyConfig(server.ClientConfig)
	config.QPS = 500
	config.Burst = 500
	testClient := clientset.NewForConfigOrDie(config)

	informers := informers.NewSharedInformerFactory(testClient, time.Second)
	return testClient, informers
}

func createNodes(t *testing.T, client *clientset.Clientset, startIndex, endIndex int) {
	var wg sync.WaitGroup
	errs := make(chan error, endIndex-startIndex)
	for i := startIndex; i < endIndex; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			node := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: fmt.Sprintf("node-%d", idx),
				},
			}
			_, err := client.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})
			if err != nil {
				errs <- err
			}
		}(i)
	}

	go func() { // wait in another go-routine to close channel
		wg.Wait()
		close(errs)
	}()

	for err := range errs {
		if err != nil {
			t.Fatalf("Failed to create node: %v", err)
		}
	}
}

func deleteNodes(t *testing.T, client *clientset.Clientset, startIndex, endIndex int) {
	var wg sync.WaitGroup
	errs := make(chan error, endIndex-startIndex)
	for i := startIndex; i < endIndex; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			name := fmt.Sprintf("node-%d", idx)
			err := client.CoreV1().Nodes().Delete(context.TODO(), name, metav1.DeleteOptions{})
			if err != nil {
				errs <- err
			}
		}(i)
	}

	go func() { // wait in another go-routine to close channel
		wg.Wait()
		close(errs)
	}()

	for err := range errs {
		if err != nil {
			t.Fatalf("Failed to delete node: %v", err)
		}
	}
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
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	testClient, informers := createClientAndInformers(t, server)
	nodeInformer := informers.Core().V1().Nodes()
	_, ctx := ktesting.NewTestContext(t)
	ttlc := ttl.NewTTLController(ctx, nodeInformer, testClient)

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go nodeInformer.Informer().Run(ctx.Done())
	go ttlc.Run(ctx, 1)

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
