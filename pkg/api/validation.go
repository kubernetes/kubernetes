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

func isSupportedManifestVersion(value string) bool {
	switch value {
	case "v1beta1", "v1beta2":
		return true
	}
	return false
}

func errInvalid(field string, value interface{}) error {
	return fmt.Errorf("%s is invalid: '%v'", field, value)
}

func errNotSupported(field string, value interface{}) error {
	return fmt.Errorf("%s is not supported: '%v'", field, value)
}

func errNotUnique(field string, value interface{}) error {
	return fmt.Errorf("%s is not unique: '%v'", field, value)
}

func validateVolumes(volumes []Volume) (util.StringSet, error) {
	allNames := util.StringSet{}
	for i := range volumes {
		vol := &volumes[i] // so we can set default values
		if len(vol.Name) > 63 || !util.IsDNSLabel(vol.Name) {
			return util.StringSet{}, errInvalid("Volume.Name", vol.Name)
		}
		if allNames.Has(vol.Name) {
			return util.StringSet{}, errNotUnique("Volume.Name", vol.Name)
		}
		allNames.Insert(vol.Name)
	}
	return allNames, nil
}

func validateContainers(containers []Container, volumes util.StringSet) error {
	allNames := util.StringSet{}
	for i := range containers {
		ctr := &containers[i] // so we can set default values
		if len(ctr.Name) > 63 || !util.IsDNSLabel(ctr.Name) {
			return errInvalid("Container.Name", ctr.Name)
		}
		if allNames.Has(ctr.Name) {
			return errNotUnique("Container.Name", ctr.Name)
		}
		allNames.Insert(ctr.Name)
		if len(ctr.Image) == 0 {
			return errInvalid("Container.Image", ctr.Name)
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
		return errInvalid("ContainerManifest.Version", manifest.Version)
	}
	if !isSupportedManifestVersion(manifest.Version) {
		return errNotSupported("ContainerManifest.Version", manifest.Version)
	}
	if len(manifest.ID) > 255 || !util.IsDNSSubdomain(manifest.ID) {
		return errInvalid("ContainerManifest.ID", manifest.ID)
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
