// +build !linux

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

package volume

import "k8s.io/klog"

// SetVolumeOwnership for unsupported volumes does nothing.  For convention, if an in-tree
// volume isnt actually setting ownership, it should call this function at the end of its "SetUpAt" implementation.
func SetVolumeOwnership(mounter Mounter, fsGroup *int64) error {
	klog.Infof("Warning: Skipped setting fsGroup (%v) volume at path %v.", fsGroup, mounter.GetPath())
	return nil
}
