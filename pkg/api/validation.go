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

// ValidateVolumes tests that the specified Volumes have valid data.  This
// includes checking formatting and uniqueness.
func ValidateVolumes(volumes []Volume) (util.StringSet, error) {
	allNames := util.StringSet{}
	for _, vol := range volumes {
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

// ValidateVolumeMounts tests that the specified VolumeMounts have valid data.
// This includes checking formatting and uniqueness.
func ValidateVolumeMounts(mounts []VolumeMount, volumes util.StringSet) error {
	for _, mnt := range mounts {
		if len(mnt.Name) == 0 {
			return fmt.Errorf("VolumeMount.Name is missing")
		}
		if !volumes.Has(mnt.Name) {
			return fmt.Errorf("VolumeMount.Name does not name a Volume: '%s'", mnt.Name)
		}
		if len(mnt.MountPath) == 0 {
			return fmt.Errorf("VolumeMount.MountPath is missing")
		}
	}
	return nil
}

// ValidatePorts tests that the specified Ports have valid data.  This includes
// checking formatting and uniqueness.
func ValidatePorts(ports []Port) error {
	allNames := util.StringSet{}
	allHostPorts := make(map[int]bool)
	for _, port := range ports {
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
		hostPort := port.ContainerPort // default
		if port.HostPort > 0 {
			hostPort = port.HostPort
			if !util.IsValidPortNum(port.HostPort) {
				return fmt.Errorf("Port.HostPort is invalid: %d", port.HostPort)
			}
		}
		//FIXME: should we set defaults (e.g. for host port) here?  Or let it stay 0 and default it later?
		if allHostPorts[hostPort] {
			return fmt.Errorf("Port.HostPort is not unique: %d", hostPort)
		}
		allHostPorts[hostPort] = true
		if port.Protocol != "" {
			if !isValidProtocol(port.Protocol) {
				return fmt.Errorf("Port.Protocol is invalid: '%s'", port.Protocol)
			}
		}
		//FIXME: set default protocol here?
	}
	return nil
}

// ValidateEnv tests that the specified EnvVars have valid data.  This includes
// checking formatting and uniqueness.
func ValidateEnv(vars []EnvVar) error {
	for _, ev := range vars {
		if len(ev.Name) == 0 || !util.IsCIdentifier(ev.Name) {
			return fmt.Errorf("EnvVar.Name is missing or malformed: '%s'", ev.Name)
		}
	}
	return nil
}

// ValidateContainers tests that the specified Containers have valid data.
// This includes checking formatting and uniqueness.
func ValidateContainers(containers []Container, volumes util.StringSet) error {
	allNames := util.StringSet{}
	for _, ctr := range containers {
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
		if err := ValidatePorts(ctr.Ports); err != nil {
			return err
		}
		if err := ValidateEnv(ctr.Env); err != nil {
			return err
		}
		if err := ValidateVolumeMounts(ctr.VolumeMounts, volumes); err != nil {
			return err
		}
	}
	return nil
}

// ValidateContainers tests that the specified ContainerManifest has valid
// data.  This includes checking formatting and uniqueness.
func ValidateManifest(manifest *ContainerManifest) error {
	//FIXME: Maybe check for missing, too long, and malformed individually with distinct error messages?
	if len(manifest.Version) == 0 || !isSupportedManifestVersion(manifest.Version) {
		return fmt.Errorf("ContainerManifest.Version is missing not supported: '%s'", manifest.Version)
	}
	if len(manifest.Id) > 255 || !util.IsDNSSubdomain(manifest.Id) {
		return fmt.Errorf("ContainerManifest.Id is missing or malformed: '%s'", manifest.Id)
	}
	//FIXME: does manifest.Id have uniqueness rules?  Callers problem.
	allVolumes, err := ValidateVolumes(manifest.Volumes)
	if err != nil {
		return err
	}
	if err := ValidateContainers(manifest.Containers, allVolumes); err != nil {
		return err
	}
	return nil
}
