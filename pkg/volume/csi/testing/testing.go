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

package testing

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csi"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

// NewTestPlugin creates a plugin mgr to load plugins and setup a fake client
func NewTestPlugin(t *testing.T, client *fakeclient.Clientset) (*volume.VolumePluginMgr, *volume.VolumePlugin, string) {
	tmpDir, err := utiltesting.MkTmpdir("csi-test")
	if err != nil {
		t.Fatalf("can't create temp dir: %v", err)
	}

	if client == nil {
		client = fakeclient.NewSimpleClientset()
	}

	client.Tracker().Add(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "fakeNode",
		},
		Spec: v1.NodeSpec{},
	})

	// Start informer for CSIDrivers.
	factory := informers.NewSharedInformerFactory(client, csi.CsiResyncPeriod)
	csiDriverInformer := factory.Storage().V1().CSIDrivers()
	csiDriverLister := csiDriverInformer.Lister()

	factory.Start(wait.NeverStop)
	syncedTypes := factory.WaitForCacheSync(wait.NeverStop)
	if len(syncedTypes) != 1 {
		t.Fatalf("informers are not synced")
	}
	for ty, ok := range syncedTypes {
		if !ok {
			t.Fatalf("failed to sync: %#v", ty)
		}
	}

	host := volumetest.NewFakeVolumeHostWithCSINodeName(t,
		tmpDir,
		client,
		csi.ProbeVolumePlugins(),
		"fakeNode",
		csiDriverLister,
		nil,
	)
	plugMgr := host.GetPluginMgr()

	plug, err := plugMgr.FindPluginByName(csi.CSIPluginName)
	if err != nil {
		t.Fatalf("can't find plugin %v", csi.CSIPluginName)
	}

	return plugMgr, &plug, tmpDir
}
