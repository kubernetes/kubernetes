/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"

	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/util/strings"
)

type flexVolumeDeleter struct {
	volume.MetricsNil
	spec   *volume.Spec
	plugin *flexVolumePlugin
}

var _ volume.Deleter = &flexVolumeDeleter{}

func (d *flexVolumeDeleter) GetPath() string {
	name := d.plugin.driverName
	return d.plugin.host.GetPodVolumeDir("", strings.EscapeQualifiedNameForDisk(name), d.spec.Name())
}

func (d *flexVolumeDeleter) Delete() error {
	volSource, _ := getVolumeSource(d.spec)

	name, ok := volSource.Options[kVolumeName]
	if !ok {
		return fmt.Errorf("%s is unable to find volume name for volume: %s", logPrefix(d.plugin), d.spec.Name())
	}

	call := d.plugin.NewDriverCall(deleteCmd)
	call.Append(name)
	call.AppendSpec(d.spec, d.plugin.host, nil)
	_, err := call.Run()
	if isCmdNotSupportedErr(err) {
		return (*deleterDefaults)(d).Delete()
	} else if err != nil {
		return err
	}
	return err
}