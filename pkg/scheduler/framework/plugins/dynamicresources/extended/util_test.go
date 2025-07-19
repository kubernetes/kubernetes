/*
Copyright 2025 The Kubernetes Authors.

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

package extended

import (
	"testing"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	resourcelisters "k8s.io/client-go/listers/resource/v1beta1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type fakeDRAManager struct {
	deviceClassLister *deviceClassLister
}

var _ framework.DeviceClassLister = &deviceClassLister{}

func (f *fakeDRAManager) ResourceClaims() framework.ResourceClaimTracker {
	return nil
}

func (f *fakeDRAManager) ResourceSlices() framework.ResourceSliceLister {
	return nil
}

func (f *fakeDRAManager) DeviceClasses() framework.DeviceClassLister {
	return f.deviceClassLister
}

type deviceClassLister struct {
	classLister resourcelisters.DeviceClassLister
}

func (l *deviceClassLister) Get(className string) (*resourceapi.DeviceClass, error) {
	return l.classLister.Get(className)
}

func (l *deviceClassLister) List() ([]*resourceapi.DeviceClass, error) {
	return l.classLister.List(labels.Everything())
}

func TestDeviceclassMapping(t *testing.T) {
	tCtx := ktesting.Init(t)

	ern1 := "example.com/gpu1"
	class1 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "class1",
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: &ern1,
		},
	}
	ern2 := "example.com/gpu2"
	class2 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "class2",
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: &ern2,
		},
	}

	client := fake.NewSimpleClientset(class1, class2)

	informerFactory := informers.NewSharedInformerFactory(client, 0)
	draManager := &fakeDRAManager{
		deviceClassLister: &deviceClassLister{classLister: informerFactory.Resource().V1beta1().DeviceClasses().Lister()},
	}

	informerFactory.Start(tCtx.Done())
	t.Cleanup(func() {
		// Need to cancel before waiting for the shutdown.
		tCtx.Cancel("test is done")
		// Now we can wait for all goroutines to stop.
		informerFactory.Shutdown()
	})

	informerFactory.WaitForCacheSync(tCtx.Done())

	rm, err := DeviceclassMapping(draManager)
	require.NoError(t, err, "calling DeviceclassMapping")
	c, ok := rm[v1.ResourceName(ern1)]
	if !ok {
		t.Errorf("result does not contain extended resource %s", ern1)
	}
	if c != "class1" {
		t.Errorf("result does not match device class name %s", "class1")
	}
	c, ok = rm[v1.ResourceName(ern2)]
	if !ok {
		t.Errorf("result does not contain extended resource %s", ern2)
	}
	if c != "class2" {
		t.Errorf("result does not match device class name %s", "class2")
	}
}
func TestNoDeviceclassMapping(t *testing.T) {
	tCtx := ktesting.Init(t)

	client := fake.NewSimpleClientset()

	informerFactory := informers.NewSharedInformerFactory(client, 0)
	draManager := &fakeDRAManager{
		deviceClassLister: &deviceClassLister{classLister: informerFactory.Resource().V1beta1().DeviceClasses().Lister()},
	}

	informerFactory.Start(tCtx.Done())
	t.Cleanup(func() {
		// Need to cancel before waiting for the shutdown.
		tCtx.Cancel("test is done")
		// Now we can wait for all goroutines to stop.
		informerFactory.Shutdown()
	})

	informerFactory.WaitForCacheSync(tCtx.Done())

	rm, err := DeviceclassMapping(draManager)
	require.NoError(t, err, "calling DeviceclassMapping")
	if len(rm) != 0 {
		t.Errorf("result should not contain extended resource")
	}
}
