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

package expand

import (
	"testing"

	"k8s.io/client-go/informers"
	"k8s.io/kubernetes/pkg/controller"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/attachdetach/testing"
)

func TestSyncHandler(t *testing.T) {
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	pvcInformer := informerFactory.Core().V1().PersistentVolumeClaims()
	pvInformer := informerFactory.Core().V1().PersistentVolumes()
	storageClassInformer := informerFactory.Storage().V1().StorageClasses()

	expc, err := NewExpandController(fakeKubeClient, pvcInformer, pvInformer, storageClassInformer, nil, nil)
	if err != nil {
		t.Fatalf("error creating expand controller : %v", err)
	}

	var expController *expandController
	expController, _ = expc.(*expandController)

	err = expController.syncHandler("default/foo")
	if err != nil {
		t.Fatalf("error running sync handler : %v", err)
	}

}
