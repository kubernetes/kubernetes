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
	"fmt"
	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/types"
)

func NewFakeNodeStatusUpdater(returnError bool) NodeStatusUpdater {
	return &fakeNodeStatusUpdater{
		returnError: returnError,
	}
}

type fakeNodeStatusUpdater struct {
	returnError bool
}

func (fnsu *fakeNodeStatusUpdater) UpdateNodeStatuses(logger klog.Logger) error {
	if fnsu.returnError {
		return fmt.Errorf("fake error on update node status")
	}

	return nil
}

func (fnsu *fakeNodeStatusUpdater) UpdateNodeStatusForNode(logger klog.Logger, nodeName types.NodeName) error {
	if fnsu.returnError {
		return fmt.Errorf("fake error on update node status")
	}

	return nil
}
