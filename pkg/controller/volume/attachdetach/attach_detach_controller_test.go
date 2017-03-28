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

package attachdetach

import (
	"testing"
	"time"

	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	"k8s.io/kubernetes/pkg/controller"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/attachdetach/testing"
)

func Test_NewAttachDetachController_Positive(t *testing.T) {
	// Arrange
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())

	// Act
	_, err := NewAttachDetachController(
		fakeKubeClient,
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().PersistentVolumes(),
		nil, /* cloud */
		nil, /* plugins */
		false,
		time.Second*5)

	// Assert
	if err != nil {
		t.Fatalf("Run failed with error. Expected: <no error> Actual: <%v>", err)
	}
}
