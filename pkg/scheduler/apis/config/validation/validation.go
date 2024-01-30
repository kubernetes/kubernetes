/*
Copyright 2018 The Kubernetes Authors.

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

package validation

import (
	"fmt"
	"reflect"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	componentbasevalidation "k8s.io/component-base/config/validation"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
)

// ValidateKubeSchedulerConfiguration ensures validation of the KubeSchedulerConfiguration struct
func ValidateKubeSchedulerConfiguration(cc *config.KubeSchedulerConfiguration) utilerrors.Aggregate {
	var errs []error
	errs = append(errs, componentbasevalidation.ValidateClientConnectionConfiguration(&cc.ClientConnection, field.NewPath("clientConnection")).ToAggregate())
	errs = append(errs, componentbasevalidation.ValidateLeaderElectionConfiguration(&cc.LeaderElection, field.NewPath("leaderElection")).ToAggregate())

	// TODO: This can be removed when ResourceLock is not available
	// Only ResourceLock values with leases are allowed
	if cc.LeaderElection.LeaderElect && cc.LeaderElection.ResourceLock != "leases" {
		leaderElectionPath := field.NewPath("leaderElection")
		errs = append(errs, field.Invalid(leaderElectionPath.Child("resourceLock"), cc.LeaderElection.ResourceLock, `resourceLock value must be "leases"`))
	}

	profilesPath := field.NewPath("profiles")
	if cc.Parallelism <= 0 {
		errs = append(errs, field.Invalid(field.NewPath("parallelism"), cc.Parallelism, "should be an integer value greater than zero"))
	}

	if len(cc.Profiles) == 0 {
		errs = append(errs, field.Required(profilesPath, ""))
	} else {
		existingProfiles := make(map[string]int, len(cc.Profiles))
		for i := range cc.Profiles {
			profile := &cc.Profiles[i]
			path := profilesPath.Index(i)
			errs = append(errs, validateKubeSchedulerProfile(path, cc.APIVersion, profile)...)
			if idx, ok := existingProfiles[profile.SchedulerName]; ok {
				errs = append(errs, field.Duplicate(path.Child("schedulerName"), profilesPath.Index(idx).Child("schedulerName")))
			}
			existingProfiles[profile.SchedulerName] = i
		}
		errs = append(errs, validateCommonQueueSort(profilesPath, cc.Profiles)...)
	}

	errs = append(errs, validatePercentageOfNodesToScore(field.NewPath("percentageOfNodesToScore"), cc.PercentageOfNodesToScore))

	if cc.PodInitialBackoffSeconds <= 0 {
		errs = append(errs, field.Invalid(field.NewPath("podInitialBackoffSeconds"),
			cc.PodInitialBackoffSeconds, "must be greater than 0"))
	}
	if cc.PodMaxBackoffSeconds < cc.PodInitialBackoffSeconds {
		errs = append(errs, field.Invalid(field.NewPath("podMaxBackoffSeconds"),
			cc.PodMaxBackoffSeconds, "must be greater than or equal to PodInitialBackoffSeconds"))
	}

	errs = append(errs, validateExtenders(field.NewPath("extenders"), cc.Extenders)...)
	return utilerrors.Flatten(utilerrors.NewAggregate(errs))
}

func validatePercentageOfNodesToScore(path *field.Path, percentageOfNodesToScore *int32) error {
	if percentageOfNodesToScore != nil {
		if *percentageOfNodesToScore < 0 || *percentageOfNodesToScore > 100 {
			return field.Invalid(path, *percentageOfNodesToScore, "not in valid range [0-100]")
		}
	}
	return nil
}

type invalidPlugins struct {
	schemeGroupVersion string
	plugins            []string
}

// invalidPluginsByVersion maintains a list of removed/deprecated plugins in each version.
// Remember to add an entry to that list when creating a new component config
// version (even if the list of invalid plugins is empty).
var invalidPluginsByVersion = []invalidPlugins{
	{
		schemeGroupVersion: v1.SchemeGroupVersion.String(),
		plugins:            []string{},
	},
}

// isPluginInvalid checks if a given plugin was removed/deprecated in the given component
// config version or earlier.
func isPluginInvalid(apiVersion string, name string) (bool, string) {
	for _, dp := range invalidPluginsByVersion {
		for _, plugin := range dp.plugins {
			if name == plugin {
				return true, dp.schemeGroupVersion
			}
		}
		if apiVersion == dp.schemeGroupVersion {
			break
		}
	}
	return false, ""
}

func validatePluginSetForInvalidPlugins(path *field.Path, apiVersion string, ps config.PluginSet) []error {
	var errs []error
	for i, plugin := range ps.Enabled {
		if invalid, invalidVersion := isPluginInvalid(apiVersion, plugin.Name); invalid {
			errs = append(errs, field.Invalid(path.Child("enabled").Index(i), plugin.Name, fmt.Sprintf("was invalid in version %q (KubeSchedulerConfiguration is version %q)", invalidVersion, apiVersion)))
		}
	}
	return errs
}

func validateKubeSchedulerProfile(path *field.Path, apiVersion string, profile *config.KubeSchedulerProfile) []error {
	var errs []error
	if len(profile.SchedulerName) == 0 {
		errs = append(errs, field.Required(path.Child("schedulerName"), ""))
	}
	errs = append(errs, validatePercentageOfNodesToScore(path.Child("percentageOfNodesToScore"), profile.PercentageOfNodesToScore))
	errs = append(errs, validatePluginConfig(path, apiVersion, profile)...)
	return errs
}

func validatePluginConfig(path *field.Path, apiVersion string, profile *config.KubeSchedulerProfile) []error {
	var errs []error
	m := map[string]interface{}{
		"DefaultPreemption":               ValidateDefaultPreemptionArgs,
		"InterPodAffinity":                ValidateInterPodAffinityArgs,
		"NodeAffinity":                    ValidateNodeAffinityArgs,
		"NodeResourcesBalancedAllocation": ValidateNodeResourcesBalancedAllocationArgs,
		"NodeResourcesFitArgs":            ValidateNodeResourcesFitArgs,
		"PodTopologySpread":               ValidatePodTopologySpreadArgs,
		"VolumeBinding":                   ValidateVolumeBindingArgs,
	}

	if profile.Plugins != nil {
		stagesToPluginSet := map[string]config.PluginSet{
			"preEnqueue": profile.Plugins.PreEnqueue,
			"queueSort":  profile.Plugins.QueueSort,
			"preFilter":  profile.Plugins.PreFilter,
			"filter":     profile.Plugins.Filter,
			"postFilter": profile.Plugins.PostFilter,
			"preScore":   profile.Plugins.PreScore,
			"score":      profile.Plugins.Score,
			"reserve":    profile.Plugins.Reserve,
			"permit":     profile.Plugins.Permit,
			"preBind":    profile.Plugins.PreBind,
			"bind":       profile.Plugins.Bind,
			"postBind":   profile.Plugins.PostBind,
		}

		pluginsPath := path.Child("plugins")
		for s, p := range stagesToPluginSet {
			errs = append(errs, validatePluginSetForInvalidPlugins(
				pluginsPath.Child(s), apiVersion, p)...)
		}
	}

	seenPluginConfig := sets.New[string]()

	for i := range profile.PluginConfig {
		pluginConfigPath := path.Child("pluginConfig").Index(i)
		name := profile.PluginConfig[i].Name
		args := profile.PluginConfig[i].Args
		if seenPluginConfig.Has(name) {
			errs = append(errs, field.Duplicate(pluginConfigPath, name))
		} else {
			seenPluginConfig.Insert(name)
		}
		if invalid, invalidVersion := isPluginInvalid(apiVersion, name); invalid {
			errs = append(errs, field.Invalid(pluginConfigPath, name, fmt.Sprintf("was invalid in version %q (KubeSchedulerConfiguration is version %q)", invalidVersion, apiVersion)))
		} else if validateFunc, ok := m[name]; ok {
			// type mismatch, no need to validate the `args`.
			if reflect.TypeOf(args) != reflect.ValueOf(validateFunc).Type().In(1) {
				errs = append(errs, field.Invalid(pluginConfigPath.Child("args"), args, "has to match plugin args"))
			} else {
				in := []reflect.Value{reflect.ValueOf(pluginConfigPath.Child("args")), reflect.ValueOf(args)}
				res := reflect.ValueOf(validateFunc).Call(in)
				// It's possible that validation function return a Aggregate, just append here and it will be flattened at the end of CC validation.
				if res[0].Interface() != nil {
					errs = append(errs, res[0].Interface().(error))
				}
			}
		}
	}
	return errs
}

func validateCommonQueueSort(path *field.Path, profiles []config.KubeSchedulerProfile) []error {
	var errs []error
	var canon config.PluginSet
	var queueSortName string
	var queueSortArgs runtime.Object
	if profiles[0].Plugins != nil {
		canon = profiles[0].Plugins.QueueSort
		if len(profiles[0].Plugins.QueueSort.Enabled) != 0 {
			queueSortName = profiles[0].Plugins.QueueSort.Enabled[0].Name
		}
		length := len(profiles[0].Plugins.QueueSort.Enabled)
		if length > 1 {
			errs = append(errs, field.Invalid(path.Index(0).Child("plugins", "queueSort", "Enabled"), length, "only one queue sort plugin can be enabled"))
		}
	}
	for _, cfg := range profiles[0].PluginConfig {
		if len(queueSortName) > 0 && cfg.Name == queueSortName {
			queueSortArgs = cfg.Args
		}
	}
	for i := 1; i < len(profiles); i++ {
		var curr config.PluginSet
		if profiles[i].Plugins != nil {
			curr = profiles[i].Plugins.QueueSort
		}
		if !apiequality.Semantic.DeepEqual(canon, curr) {
			errs = append(errs, field.Invalid(path.Index(i).Child("plugins", "queueSort"), curr, "queueSort must be the same for all profiles"))
		}
		for _, cfg := range profiles[i].PluginConfig {
			if cfg.Name == queueSortName && !apiequality.Semantic.DeepEqual(queueSortArgs, cfg.Args) {
				errs = append(errs, field.Invalid(path.Index(i).Child("pluginConfig", "args"), cfg.Args, "queueSort must be the same for all profiles"))
			}
		}
	}
	return errs
}

// validateExtenders validates the configured extenders for the Scheduler
func validateExtenders(fldPath *field.Path, extenders []config.Extender) []error {
	var errs []error
	binders := 0
	extenderManagedResources := sets.New[string]()
	for i, extender := range extenders {
		path := fldPath.Index(i)
		if len(extender.PrioritizeVerb) > 0 && extender.Weight <= 0 {
			errs = append(errs, field.Invalid(path.Child("weight"),
				extender.Weight, "must have a positive weight applied to it"))
		}
		if extender.BindVerb != "" {
			binders++
		}
		for j, resource := range extender.ManagedResources {
			managedResourcesPath := path.Child("managedResources").Index(j)
			validationErrors := validateExtendedResourceName(managedResourcesPath.Child("name"), v1.ResourceName(resource.Name))
			errs = append(errs, validationErrors...)
			if extenderManagedResources.Has(resource.Name) {
				errs = append(errs, field.Invalid(managedResourcesPath.Child("name"),
					resource.Name, "duplicate extender managed resource name"))
			}
			extenderManagedResources.Insert(resource.Name)
		}
	}
	if binders > 1 {
		errs = append(errs, field.Invalid(fldPath, fmt.Sprintf("found %d extenders implementing bind", binders), "only one extender can implement bind"))
	}
	return errs
}

// validateExtendedResourceName checks whether the specified name is a valid
// extended resource name.
func validateExtendedResourceName(path *field.Path, name v1.ResourceName) []error {
	var validationErrors []error
	for _, msg := range validation.IsQualifiedName(string(name)) {
		validationErrors = append(validationErrors, field.Invalid(path, name, msg))
	}
	if len(validationErrors) != 0 {
		return validationErrors
	}
	if !v1helper.IsExtendedResourceName(name) {
		validationErrors = append(validationErrors, field.Invalid(path, string(name), "is an invalid extended resource name"))
	}
	return validationErrors
}
