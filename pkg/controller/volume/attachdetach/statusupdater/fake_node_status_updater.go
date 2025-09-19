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

package statusupdater

import (
	"context"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
)

// Passing nil to simulate update failure
func NewFakeNodeStatusUpdater(asw cache.ActualStateOfWorld) NodeStatusUpdater {
	u := &fakeNodeStatusUpdater{
		asw: asw,
	}
	if asw != nil {
		asw.SetNodeUpdateHook(u.QueueUpdate)
	}
	return u
}

type fakeNodeStatusUpdater struct {
	asw cache.ActualStateOfWorld
}

func (fnsu *fakeNodeStatusUpdater) QueueUpdate(nodeName types.NodeName) {
	go func() {
		logger := klog.Background()
		_, removed := fnsu.asw.GetVolumesToReportAttachedForNode(logger, nodeName)
		fnsu.asw.ConfirmNodeStatusRemoved(logger, nodeName, removed)
	}()
}

func (fnsu *fakeNodeStatusUpdater) Run(ctx context.Context, workers int) {
	<-ctx.Done()
}
