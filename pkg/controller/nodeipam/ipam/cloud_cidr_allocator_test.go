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

package ipam

import (
	"fmt"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
)

func hasNodeInProcessing(ca *cloudCIDRAllocator, name string) bool {
	ca.lock.Lock()
	defer ca.lock.Unlock()

	_, found := ca.nodesInProcessing[name]
	return found
}

func TestBoundedRetries(t *testing.T) {
	clientSet := fake.NewSimpleClientset()
	updateChan := make(chan string, 1) // need to buffer as we are using only on go routine
	stopChan := make(chan struct{})
	sharedInfomer := informers.NewSharedInformerFactory(clientSet, 1*time.Hour)
	ca := &cloudCIDRAllocator{
		client:            clientSet,
		nodeUpdateChannel: updateChan,
		nodeLister:        sharedInfomer.Core().V1().Nodes().Lister(),
		nodesSynced:       sharedInfomer.Core().V1().Nodes().Informer().HasSynced,
		nodesInProcessing: map[string]*nodeProcessingInfo{},
	}
	go ca.worker(stopChan)
	nodeName := "testNode"
	ca.AllocateOrOccupyCIDR(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: nodeName,
		},
	})
	for hasNodeInProcessing(ca, nodeName) {
		// wait for node to finish processing (should terminate and not time out)
	}
}

func withinExpectedRange(got time.Duration, expected time.Duration) bool {
	return got >= expected/2 && got <= 3*expected/2
}

func TestNodeUpdateRetryTimeout(t *testing.T) {
	for _, tc := range []struct {
		count int
		want  time.Duration
	}{
		{count: 0, want: 250 * time.Millisecond},
		{count: 1, want: 500 * time.Millisecond},
		{count: 2, want: 1000 * time.Millisecond},
		{count: 3, want: 2000 * time.Millisecond},
		{count: 50, want: 5000 * time.Millisecond},
	} {
		t.Run(fmt.Sprintf("count %d", tc.count), func(t *testing.T) {
			if got := nodeUpdateRetryTimeout(tc.count); !withinExpectedRange(got, tc.want) {
				t.Errorf("nodeUpdateRetryTimeout(tc.count) = %v; want %v", got, tc.want)
			}
		})
	}
}
