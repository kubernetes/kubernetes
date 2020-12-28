/*
Copyright 2020 The Kubernetes Authors.

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

package config

import (
	"fmt"
	"io/ioutil"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	util "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	internal "k8s.io/controller-manager/config"
	"k8s.io/controller-manager/config/v1alpha1"
)

// ResourceLockLeases is the resourceLock value for 'leases' API
const ResourceLockLeases = "leases"

// ResourceLockEndpoints is the resourceLock value for 'endpoints' API
const ResourceLockEndpoints = "endpoints"

var cfgScheme = runtime.NewScheme()

func init() {
	// internal
	util.Must(internal.AddToScheme(cfgScheme))

	// v1alpha1
	util.Must(v1alpha1.AddToScheme(cfgScheme))
	util.Must(cfgScheme.SetVersionPriority(v1alpha1.SchemeGroupVersion))
}

// ReadLeaderMigrationConfiguration reads LeaderMigrationConfiguration from a YAML file at the given path.
// The parsed LeaderMigrationConfiguration may be invalid.
// It returns an error if the file did not exist.
func ReadLeaderMigrationConfiguration(configFilePath string) (*internal.LeaderMigrationConfiguration, error) {
	data, err := ioutil.ReadFile(configFilePath)
	if err != nil {
		return nil, fmt.Errorf("unable to read leader migration configuration from %q: %v", configFilePath, err)
	}
	config, gvk, err := serializer.NewCodecFactory(cfgScheme).UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, err
	}
	internalConfig, ok := config.(*internal.LeaderMigrationConfiguration)
	if !ok {
		return nil, fmt.Errorf("unexpected config type: %v", gvk)
	}
	return internalConfig, nil
}

// ValidateLeaderMigrationConfiguration validates the LeaderMigrationConfiguration against common errors.
// It checks required names and whether resourceLock is either 'leases' or 'endpoints'.
// It will return nil if it does not find anything wrong.
func ValidateLeaderMigrationConfiguration(config *internal.LeaderMigrationConfiguration) (allErrs field.ErrorList) {
	if config.LeaderName == "" {
		allErrs = append(allErrs, field.Required(field.NewPath("leaderName"),
			"leaderName must be set for LeaderMigrationConfiguration"))
	}
	if config.ResourceLock != ResourceLockLeases && config.ResourceLock != ResourceLockEndpoints {
		allErrs = append(allErrs, field.Invalid(field.NewPath("resourceLock"), config.ResourceLock,
			"resource Lock must be one of 'leases' or 'endpoints'"))
	}
	// validate controllerLeaders
	fldPath := field.NewPath("controllerLeaders")
	for i, controllerLeader := range config.ControllerLeaders {
		path := fldPath.Index(i)
		allErrs = append(allErrs, validateControllerLeaderConfiguration(path, &controllerLeader)...)
	}
	return
}

func validateControllerLeaderConfiguration(path *field.Path, config *internal.ControllerLeaderConfiguration) (allErrs field.ErrorList) {
	if config == nil {
		return
	}
	if config.Component == "" {
		allErrs = append(allErrs, field.Required(path.Child("component"), "component must be set"))
	}
	if config.Name == "" {
		allErrs = append(allErrs, field.Required(path.Child("name"), "name must be set"))
	}
	return
}
