/*
Copyright 2014 Google Inc. All rights reserved.

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

package api

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

func isValidProtocol(proto string) bool {
	if proto == "TCP" || proto == "UDP" {
		return true
	}
	return false
}

func isSupportedManifestVersion(value string) bool {
	switch value {
	case "v1beta1":
		return true
	}
	return false
}

func validateVolumes(volumes []Volume) (util.StringSet, error) {
	allNames := util.StringSet{}
	for i := range volumes {
		vol := &volumes[i] // so we can set default values
		if len(vol.Name) > 63 || !util.IsDNSLabel(vol.Name) {
			return util.StringSet{}, fmt.Errorf("Volume.Name is missing or malformed: '%s'", vol.Name)
		}
		if allNames.Has(vol.Name) {
			return util.StringSet{}, fmt.Errorf("Volume.Name is not unique: '%s'", vol.Name)
		}
		allNames.Insert(vol.Name)
	}
	return allNames, nil
}

func validateVolumeMounts(mounts []VolumeMount, volumes util.StringSet) error {
	for i := range mounts {
		mnt := &mounts[i] // so we can set default values
		if len(mnt.Name) == 0 {
			return fmt.Errorf("VolumeMount.Name is missing")
		}
		if !volumes.Has(mnt.Name) {
			return fmt.Errorf("VolumeMount.Name does not name a Volume: '%s'", mnt.Name)
		}
		if len(mnt.MountPath) == 0 {
			// Backwards compat.
			if len(mnt.Path) == 0 {
				return fmt.Errorf("VolumeMount.MountPath is missing")
			}
			glog.Warning("DEPRECATED: VolumeMount.Path has been replaced by VolumeMount.MountPath")
			mnt.MountPath = mnt.Path
			mnt.Path = ""
		}
	}
	return nil
}

func validatePorts(ports []Port) error {
	allNames := util.StringSet{}
	for i := range ports {
		port := &ports[i] // so we can set default values
		if port.Name != "" {
			if len(port.Name) > 63 || !util.IsDNSLabel(port.Name) {
				return fmt.Errorf("Port.Name is missing or malformed: '%s'", port.Name)
			}
			if allNames.Has(port.Name) {
				return fmt.Errorf("Port.Name is not unique: '%s'", port.Name)
			}
			allNames.Insert(port.Name)
		}
		if !util.IsValidPortNum(port.ContainerPort) {
			return fmt.Errorf("Port.ContainerPort is missing or invalid: %d", port.ContainerPort)
		}
		if port.HostPort == 0 {
			port.HostPort = port.ContainerPort
		} else if !util.IsValidPortNum(port.HostPort) {
			return fmt.Errorf("Port.HostPort is invalid: %d", port.HostPort)
		}
		if port.Protocol == "" {
			port.Protocol = "TCP"
		} else if !isValidProtocol(port.Protocol) {
			return fmt.Errorf("Port.Protocol is invalid: '%s'", port.Protocol)
		}
	}
	return nil
}

func validateEnv(vars []EnvVar) error {
	for i := range vars {
		ev := &vars[i] // so we can set default values
		if len(ev.Name) == 0 || !util.IsCIdentifier(ev.Name) {
			return fmt.Errorf("EnvVar.Name is missing or malformed: '%s'", ev.Name)
		}
	}
	return nil
}

// Runs an extraction function on each Port of each Container, accumulating the
// results and returning an error if any ports conflict.
func AccumulateUniquePorts(containers []Container, all map[int]bool, extract func(*Port) int) error {
	for ci := range containers {
		ctr := &containers[ci]
		for pi := range ctr.Ports {
			port := extract(&ctr.Ports[pi])
			if all[port] {
				return fmt.Errorf("port %d is already in use", port)
			}
			all[port] = true
		}
	}
	return nil
}

// Checks for colliding Port.HostPort values across a slice of containers.
func checkHostPortConflicts(containers []Container) error {
	allPorts := map[int]bool{}
	return AccumulateUniquePorts(containers, allPorts, func(p *Port) int { return p.HostPort })
}

func validateContainers(containers []Container, volumes util.StringSet) error {
	allNames := util.StringSet{}
	for i := range containers {
		ctr := &containers[i] // so we can set default values
		if len(ctr.Name) > 63 || !util.IsDNSLabel(ctr.Name) {
			return fmt.Errorf("Container.Name is missing or malformed: '%s'", ctr.Name)
		}
		if allNames.Has(ctr.Name) {
			return fmt.Errorf("Container.Name is not unique: '%s'", ctr.Name)
		}
		allNames.Insert(ctr.Name)
		if len(ctr.Image) == 0 {
			return fmt.Errorf("Container.Image is missing")
		}
		if err := validatePorts(ctr.Ports); err != nil {
			return err
		}
		if err := validateEnv(ctr.Env); err != nil {
			return err
		}
		if err := validateVolumeMounts(ctr.VolumeMounts, volumes); err != nil {
			return err
		}
	}
	// Check for colliding ports across all containers.
	// TODO(thockin): This really is dependent on the network config of the host (IP per pod?)
	// and the config of the new manifest.  But we have not specced that out yet, so we'll just
	// make some assumptions for now.  As of now, pods share a network namespace, which means that
	// every Port.HostPort across the whole pod must be unique.
	return checkHostPortConflicts(containers)
}

// ValidateManifest tests that the specified ContainerManifest has valid data.
// This includes checking formatting and uniqueness.  It also canonicalizes the
// structure by setting default values and implementing any backwards-compatibility
// tricks.
func ValidateManifest(manifest *ContainerManifest) error {
	if len(manifest.Version) == 0 || !isSupportedManifestVersion(manifest.Version) {
		return fmt.Errorf("ContainerManifest.Version is missing not supported: '%s'", manifest.Version)
	}
	if len(manifest.Id) > 255 || !util.IsDNSSubdomain(manifest.Id) {
		return fmt.Errorf("ContainerManifest.Id is missing or malformed: '%s'", manifest.Id)
	}
	allVolumes, err := validateVolumes(manifest.Volumes)
	if err != nil {
		return err
	}
	if err := validateContainers(manifest.Containers, allVolumes); err != nil {
		return err
	}
	return nil
}
