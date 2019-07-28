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
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csi"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"testing"
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

	// Start informer for CSIDrivers.
	factory := informers.NewSharedInformerFactory(client, csi.CsiResyncPeriod)
	csiDriverInformer := factory.Storage().V1beta1().CSIDrivers()
	csiDriverLister := csiDriverInformer.Lister()
	go factory.Start(wait.NeverStop)

	host := volumetest.NewFakeVolumeHostWithCSINodeName(
		tmpDir,
		client,
		nil,
		"fakeNode",
		csiDriverLister,
	)
	plugMgr := &volume.VolumePluginMgr{}
	plugMgr.InitPlugins(csi.ProbeVolumePlugins(), nil /* prober */, host)

	plug, err := plugMgr.FindPluginByName(csi.CSIPluginName)
	if err != nil {
		t.Fatalf("can't find plugin %v", csi.CSIPluginName)
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.CSIDriverRegistry) {
		// Wait until the informer in CSI volume plugin has all CSIDrivers.
		wait.PollImmediate(csi.TestInformerSyncPeriod, csi.TestInformerSyncTimeout, func() (bool, error) {
			return csiDriverInformer.Informer().HasSynced(), nil
		})
	}

	return plugMgr, &plug, tmpDir
}
