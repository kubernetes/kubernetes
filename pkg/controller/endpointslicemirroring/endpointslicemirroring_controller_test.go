/*
Copyright 2019 The Kubernetes Authors.

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

package endpointslice

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/controller"
)

// Most of the tests related to EndpointSlice allocation can be found in reconciler_test.go
// Tests here primarily focus on unique controller functionality before the reconciler begins

var alwaysReady = func() bool { return true }

type endpointSliceMirroringController struct {
	*Controller
	endpointSliceStore cache.Store
	endpointsStore     cache.Store
}

func newController(nodeNames []string, batchPeriod time.Duration) (*fake.Clientset, *endpointSliceMirroringController) {
	client := newClientset()
	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())

	esController := NewController(
		informerFactory.Core().V1().Endpoints(),
		informerFactory.Discovery().V1beta1().EndpointSlices(),
		int32(100),
		client,
		batchPeriod)

	esController.endpointsSynced = alwaysReady
	esController.endpointSlicesSynced = alwaysReady

	return client, &endpointSliceMirroringController{
		esController,
		informerFactory.Core().V1().Endpoints().Informer().GetStore(),
		informerFactory.Discovery().V1beta1().EndpointSlices().Informer().GetStore(),
	}
}

// Ensure SyncEndpoints works with an empty Endpoints resource.
func TestSyncEndpointsEmpty(t *testing.T) {
	ns := metav1.NamespaceDefault
	endpointsName := "testing-1"
	client, esController := newController([]string{"node-1"}, time.Duration(0))
	esController.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: endpointsName, Namespace: ns},
		Subsets: []v1.EndpointSubset{{
			Ports: []v1.EndpointPort{{Port: 80}},
		}},
	})

	err := esController.syncEndpoints(fmt.Sprintf("%s/%s", ns, endpointsName))
	assert.Nil(t, err)
	assert.Len(t, client.Actions(), 0)
}
