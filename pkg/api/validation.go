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

var (
	supportedManifestVersions = util.NewStringSet("v1beta1", "v1beta2")
)

// ValidationErrorEnum is a type of validation error.
type ValidationErrorEnum string

// These are known errors of validation.
const (
	ErrTypeInvalid      ValidationErrorEnum = "invalid value"
	ErrTypeNotSupported ValidationErrorEnum = "unsupported value"
	ErrTypeDuplicate    ValidationErrorEnum = "duplicate value"
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

func validateVolumes(volumes []Volume) (util.StringSet, error) {
	allNames := util.StringSet{}
	for i := range volumes {
		vol := &volumes[i] // so we can set default values
		if len(vol.Name) > 63 || !util.IsDNSLabel(vol.Name) {
			return util.StringSet{}, makeInvalidError("Volume.Name", vol.Name)
		}
		if allNames.Has(vol.Name) {
			return util.StringSet{}, makeDuplicateError("Volume.Name", vol.Name)
		}
		allNames.Insert(vol.Name)
	}
	return allNames, nil
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

func validateContainers(containers []Container, volumes util.StringSet) error {
	allNames := util.StringSet{}
	for i := range containers {
		ctr := &containers[i] // so we can set default values
		if len(ctr.Name) > 63 || !util.IsDNSLabel(ctr.Name) {
			return makeInvalidError("Container.Name", ctr.Name)
		}
		if allNames.Has(ctr.Name) {
			return makeDuplicateError("Container.Name", ctr.Name)
		}
		allNames.Insert(ctr.Name)
		if len(ctr.Image) == 0 {
			return makeInvalidError("Container.Image", ctr.Name)
		}
		if err := validateEnv(ctr.Env); err != nil {
			return err
		}

		// TODO(thockin): finish validation.
	}
	return nil
}

// ValidateManifest tests that the specified ContainerManifest has valid data.
// This includes checking formatting and uniqueness.  It also canonicalizes the
// structure by setting default values and implementing any backwards-compatibility
// tricks.
// TODO(thockin): We should probably collect all errors rather than aborting the validation.
func ValidateManifest(manifest *ContainerManifest) error {
	if len(manifest.Version) == 0 {
		return makeInvalidError("ContainerManifest.Version", manifest.Version)
	}
	if !supportedManifestVersions.Has(manifest.Version) {
		return makeNotSupportedError("ContainerManifest.Version", manifest.Version)
	}
	if len(manifest.ID) > 255 || !util.IsDNSSubdomain(manifest.ID) {
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
