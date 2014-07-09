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
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

// ValidationErrorEnum is a type of validation error.
type ValidationErrorEnum string

// These are known errors of validation.
const (
	ErrTypeInvalid      ValidationErrorEnum = "invalid value"
	ErrTypeNotSupported ValidationErrorEnum = "unsupported value"
	ErrTypeDuplicate    ValidationErrorEnum = "duplicate value"
	ErrTypeNotFound     ValidationErrorEnum = "not found"
)

// ValidationError is an implementation of the 'error' interface, which represents an error of validation.
type ValidationError struct {
	ErrorType  ValidationErrorEnum
	ErrorField string
	BadValue   interface{}
}

func (v ValidationError) Error() string {
	return fmt.Sprintf("%s: %v '%v'", v.ErrorField, v.ErrorType, v.BadValue)
}

// Factory functions for errors.
func makeInvalidError(field string, value interface{}) ValidationError {
	return ValidationError{ErrTypeInvalid, field, value}
}

func makeNotSupportedError(field string, value interface{}) ValidationError {
	return ValidationError{ErrTypeNotSupported, field, value}
}

func makeDuplicateError(field string, value interface{}) ValidationError {
	return ValidationError{ErrTypeDuplicate, field, value}
}

func makeNotFoundError(field string, value interface{}) ValidationError {
	return ValidationError{ErrTypeNotFound, field, value}
}

func validateVolumes(volumes []Volume) (util.StringSet, error) {
	allNames := util.StringSet{}
	for i := range volumes {
		vol := &volumes[i] // so we can set default values
		if !util.IsDNSLabel(vol.Name) {
			return util.StringSet{}, makeInvalidError("Volume.Name", vol.Name)
		}
		if allNames.Has(vol.Name) {
			return util.StringSet{}, makeDuplicateError("Volume.Name", vol.Name)
		}
		allNames.Insert(vol.Name)
	}
	return allNames, nil
}

var supportedPortProtocols util.StringSet = util.NewStringSet("TCP", "UDP")

func validatePorts(ports []Port) error {
	allNames := util.StringSet{}
	for i := range ports {
		port := &ports[i] // so we can set default values
		if len(port.Name) > 0 {
			if len(port.Name) > 63 || !util.IsDNSLabel(port.Name) {
				return makeInvalidError("Port.Name", port.Name)
			}
			if allNames.Has(port.Name) {
				return makeDuplicateError("Port.name", port.Name)
			}
			allNames.Insert(port.Name)
		}
		if !util.IsValidPortNum(port.ContainerPort) {
			return makeInvalidError("Port.ContainerPort", port.ContainerPort)
		}
		if port.HostPort == 0 {
			port.HostPort = port.ContainerPort
		} else if !util.IsValidPortNum(port.HostPort) {
			return makeInvalidError("Port.HostPort", port.HostPort)
		}
		if len(port.Protocol) == 0 {
			port.Protocol = "TCP"
		} else if !supportedPortProtocols.Has(strings.ToUpper(port.Protocol)) {
			return makeNotSupportedError("Port.Protocol", port.Protocol)
		}
	}
	return nil
}

func validateEnv(vars []EnvVar) error {
	for i := range vars {
		ev := &vars[i] // so we can set default values
		if len(ev.Name) == 0 {
			// Backwards compat.
			if len(ev.Key) == 0 {
				return makeInvalidError("EnvVar.Name", ev.Name)
			}
			glog.Warning("DEPRECATED: EnvVar.Key has been replaced by EnvVar.Name")
			ev.Name = ev.Key
			ev.Key = ""
		}
		if !util.IsCIdentifier(ev.Name) {
			return makeInvalidError("EnvVar.Name", ev.Name)
		}
	}
	return nil
}

func validateVolumeMounts(mounts []VolumeMount, volumes util.StringSet) error {
	for i := range mounts {
		mnt := &mounts[i] // so we can set default values
		if len(mnt.Name) == 0 {
			return makeInvalidError("VolumeMount.Name", mnt.Name)
		}
		if !volumes.Has(mnt.Name) {
			return makeNotFoundError("VolumeMount.Name", mnt.Name)
		}
		if len(mnt.MountPath) == 0 {
			// Backwards compat.
			if len(mnt.Path) == 0 {
				return makeInvalidError("VolumeMount.MountPath", mnt.MountPath)
			}
			glog.Warning("DEPRECATED: VolumeMount.Path has been replaced by VolumeMount.MountPath")
			mnt.MountPath = mnt.Path
			mnt.Path = ""
		}
	}
	return nil
}

// AccumulateUniquePorts runs an extraction function on each Port of each Container,
// accumulating the results and returning an error if any ports conflict.
func AccumulateUniquePorts(containers []Container, accumulator map[int]bool, extract func(*Port) int) error {
	for ci := range containers {
		ctr := &containers[ci]
		for pi := range ctr.Ports {
			port := extract(&ctr.Ports[pi])
			if accumulator[port] {
				return makeDuplicateError("Port", port)
			}
			accumulator[port] = true
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
		if !util.IsDNSLabel(ctr.Name) {
			return makeInvalidError("Container.Name", ctr.Name)
		}
		if allNames.Has(ctr.Name) {
			return makeDuplicateError("Container.Name", ctr.Name)
		}
		allNames.Insert(ctr.Name)
		if len(ctr.Image) == 0 {
			return makeInvalidError("Container.Image", ctr.Name)
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

var supportedManifestVersions util.StringSet = util.NewStringSet("v1beta1", "v1beta2")

// ValidateManifest tests that the specified ContainerManifest has valid data.
// This includes checking formatting and uniqueness.  It also canonicalizes the
// structure by setting default values and implementing any backwards-compatibility
// tricks.
// TODO(thockin): We should probably collect all errors rather than aborting the validation.
func ValidateManifest(manifest *ContainerManifest) error {
	if len(manifest.Version) == 0 {
		return makeInvalidError("ContainerManifest.Version", manifest.Version)
	}
	if !supportedManifestVersions.Has(strings.ToLower(manifest.Version)) {
		return makeNotSupportedError("ContainerManifest.Version", manifest.Version)
	}
	if !util.IsDNSSubdomain(manifest.ID) {
		return makeInvalidError("ContainerManifest.ID", manifest.ID)
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
