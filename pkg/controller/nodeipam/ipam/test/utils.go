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

package test

import (
	"net"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/testutil"
	netutils "k8s.io/utils/net"
)

const NodePollInterval = 10 * time.Millisecond

var AlwaysReady = func() bool { return true }

// MustParseCIDR returns the CIDR range parsed from s or panics if the string
// cannot be parsed.
func MustParseCIDR(s string) *net.IPNet {
	_, ret, err := netutils.ParseCIDRSloppy(s)
	if err != nil {
		panic(err)
	}
	return ret
}

// FakeNodeInformer creates a fakeNodeInformer using the provided fakeNodeHandler.
func FakeNodeInformer(fakeNodeHandler *testutil.FakeNodeHandler) coreinformers.NodeInformer {
	fakeClient := &fake.Clientset{}
	fakeInformerFactory := informers.NewSharedInformerFactory(fakeClient, controller.NoResyncPeriodFunc())
	fakeNodeInformer := fakeInformerFactory.Core().V1().Nodes()

	for _, node := range fakeNodeHandler.Existing {
		fakeNodeInformer.Informer().GetStore().Add(node)
	}

	return fakeNodeInformer
}

func WaitForUpdatedNodeWithTimeout(nodeHandler *testutil.FakeNodeHandler, number int, timeout time.Duration) error {
	return wait.Poll(NodePollInterval, timeout, func() (bool, error) {
		if len(nodeHandler.GetUpdatedNodesCopy()) >= number {
			return true, nil
		}
		return false, nil
	})
}
