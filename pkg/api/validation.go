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

// A helper for accumulating errors.  This could be moved to util if anyone else needs it.
type errorList []error

func (list *errorList) Append(errs ...error) {
	*list = append(*list, errs...)
}

func validateVolumes(volumes []Volume) (util.StringSet, errorList) {
	allErrs := errorList{}

	allNames := util.StringSet{}
	for i := range volumes {
		vol := &volumes[i] // so we can set default values
		if !util.IsDNSLabel(vol.Name) {
			allErrs.Append(makeInvalidError("Volume.Name", vol.Name))
		} else if allNames.Has(vol.Name) {
			allErrs.Append(makeDuplicateError("Volume.Name", vol.Name))
		} else {
			allNames.Insert(vol.Name)
		}
	}
	return allNames, allErrs
}

var supportedPortProtocols = util.NewStringSet("TCP", "UDP")

func validatePorts(ports []Port) errorList {
	allErrs := errorList{}

	allNames := util.StringSet{}
	for i := range ports {
		port := &ports[i] // so we can set default values
		if len(port.Name) > 0 {
			if len(port.Name) > 63 || !util.IsDNSLabel(port.Name) {
				allErrs.Append(makeInvalidError("Port.Name", port.Name))
			} else if allNames.Has(port.Name) {
				allErrs.Append(makeDuplicateError("Port.name", port.Name))
			} else {
				allNames.Insert(port.Name)
			}
		}
		if !util.IsValidPortNum(port.ContainerPort) {
			allErrs.Append(makeInvalidError("Port.ContainerPort", port.ContainerPort))
		}
		if port.HostPort == 0 {
			port.HostPort = port.ContainerPort
		} else if !util.IsValidPortNum(port.HostPort) {
			allErrs.Append(makeInvalidError("Port.HostPort", port.HostPort))
		}
		if len(port.Protocol) == 0 {
			port.Protocol = "TCP"
		} else if !supportedPortProtocols.Has(strings.ToUpper(port.Protocol)) {
			allErrs.Append(makeNotSupportedError("Port.Protocol", port.Protocol))
		}
	}
	return allErrs
}

func validateEnv(vars []EnvVar) errorList {
	allErrs := errorList{}

	for i := range vars {
		ev := &vars[i] // so we can set default values
		if len(ev.Name) == 0 {
			// Backwards compat.
			if len(ev.Key) == 0 {
				allErrs.Append(makeInvalidError("EnvVar.Name", ev.Name))
			} else {
				glog.Warning("DEPRECATED: EnvVar.Key has been replaced by EnvVar.Name")
				ev.Name = ev.Key
				ev.Key = ""
			}
		}
		if !util.IsCIdentifier(ev.Name) {
			allErrs.Append(makeInvalidError("EnvVar.Name", ev.Name))
		}
	}
	return allErrs
}

func validateVolumeMounts(mounts []VolumeMount, volumes util.StringSet) errorList {
	allErrs := errorList{}

	for i := range mounts {
		mnt := &mounts[i] // so we can set default values
		if len(mnt.Name) == 0 {
			allErrs.Append(makeInvalidError("VolumeMount.Name", mnt.Name))
		} else if !volumes.Has(mnt.Name) {
			allErrs.Append(makeNotFoundError("VolumeMount.Name", mnt.Name))
		}
		if len(mnt.MountPath) == 0 {
			// Backwards compat.
			if len(mnt.Path) == 0 {
				allErrs.Append(makeInvalidError("VolumeMount.MountPath", mnt.MountPath))
			} else {
				glog.Warning("DEPRECATED: VolumeMount.Path has been replaced by VolumeMount.MountPath")
				mnt.MountPath = mnt.Path
				mnt.Path = ""
			}
		}
	}
	return allErrs
}

// AccumulateUniquePorts runs an extraction function on each Port of each Container,
// accumulating the results and returning an error if any ports conflict.
func AccumulateUniquePorts(containers []Container, accumulator map[int]bool, extract func(*Port) int) errorList {
	allErrs := errorList{}

	for ci := range containers {
		ctr := &containers[ci]
		for pi := range ctr.Ports {
			port := extract(&ctr.Ports[pi])
			if accumulator[port] {
				allErrs.Append(makeDuplicateError("Port", port))
			} else {
				accumulator[port] = true
			}
		}
	}
	return allErrs
}

// Checks for colliding Port.HostPort values across a slice of containers.
func checkHostPortConflicts(containers []Container) errorList {
	allPorts := map[int]bool{}
	return AccumulateUniquePorts(containers, allPorts, func(p *Port) int { return p.HostPort })
}

func validateContainers(containers []Container, volumes util.StringSet) errorList {
	allErrs := errorList{}

	allNames := util.StringSet{}
	for i := range containers {
		ctr := &containers[i] // so we can set default values
		if !util.IsDNSLabel(ctr.Name) {
			allErrs.Append(makeInvalidError("Container.Name", ctr.Name))
		} else if allNames.Has(ctr.Name) {
			allErrs.Append(makeDuplicateError("Container.Name", ctr.Name))
		} else {
			allNames.Insert(ctr.Name)
		}
		if len(ctr.Image) == 0 {
			allErrs.Append(makeInvalidError("Container.Image", ctr.Name))
		}
		allErrs.Append(validatePorts(ctr.Ports)...)
		allErrs.Append(validateEnv(ctr.Env)...)
		allErrs.Append(validateVolumeMounts(ctr.VolumeMounts, volumes)...)
	}
	// Check for colliding ports across all containers.
	// TODO(thockin): This really is dependent on the network config of the host (IP per pod?)
	// and the config of the new manifest.  But we have not specced that out yet, so we'll just
	// make some assumptions for now.  As of now, pods share a network namespace, which means that
	// every Port.HostPort across the whole pod must be unique.
	allErrs.Append(checkHostPortConflicts(containers)...)

	return allErrs
}

var supportedManifestVersions = util.NewStringSet("v1beta1", "v1beta2")

// ValidateManifest tests that the specified ContainerManifest has valid data.
// This includes checking formatting and uniqueness.  It also canonicalizes the
// structure by setting default values and implementing any backwards-compatibility
// tricks.
func ValidateManifest(manifest *ContainerManifest) []error {
	allErrs := errorList{}

	if len(manifest.Version) == 0 {
		allErrs.Append(makeInvalidError("ContainerManifest.Version", manifest.Version))
	} else if !supportedManifestVersions.Has(strings.ToLower(manifest.Version)) {
		allErrs.Append(makeNotSupportedError("ContainerManifest.Version", manifest.Version))
	}
	if !util.IsDNSSubdomain(manifest.ID) {
		allErrs.Append(makeInvalidError("ContainerManifest.ID", manifest.ID))
	}
	allVolumes, errs := validateVolumes(manifest.Volumes)
	if len(errs) != 0 {
		allErrs.Append(errs...)
	}
	allErrs.Append(validateContainers(manifest.Containers, allVolumes)...)
	return []error(allErrs)
}

func ValidateService(service *Service) []error {
	allErrs := errorList{}
	if service.ID == "" {
		allErrs.Append(makeInvalidError("Service.ID", service.ID))
	}
	if len(service.Selector) == 0 {
		allErrs.Append(makeInvalidError("Service.Selector", service.Selector))
	}
	return []error(allErrs)
}
