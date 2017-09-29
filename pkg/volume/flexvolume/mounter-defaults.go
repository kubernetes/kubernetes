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

package flexvolume

import (
	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/volume"
)

type mounterDefaults flexVolumeMounter

// SetUpAt is part of the volume.Mounter interface.
// This implementation relies on the attacher's device mount path and does a bind mount to dir.
func (f *mounterDefaults) SetUpAt(dir string, fsGroup *int64) error {
	glog.Warning(logPrefix(f.plugin), "using default SetUpAt to ", dir)

	src, err := f.plugin.getDeviceMountPath(f.spec)
	if err != nil {
		return err
	}

	if err := doMount(f.mounter, src, dir, "auto", []string{"bind"}); err != nil {
		return err
	}

	return nil
}

// Returns the default volume attributes.
func (f *mounterDefaults) GetAttributes() volume.Attributes {
	glog.V(5).Infof(logPrefix(f.plugin), "using default GetAttributes")
	return volume.Attributes{
		ReadOnly:        f.readOnly,
		Managed:         !f.readOnly,
		SupportsSELinux: f.flexVolume.plugin.capabilities.SELinuxRelabel,
	}
}

func (f *mounterDefaults) CanMount() error {
	return nil
}
