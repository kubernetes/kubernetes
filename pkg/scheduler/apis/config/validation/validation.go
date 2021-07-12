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

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	componentbasevalidation "k8s.io/component-base/config/validation"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/v1beta1"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/v1beta2"
)

// ValidateKubeSchedulerConfiguration ensures validation of the KubeSchedulerConfiguration struct
func ValidateKubeSchedulerConfiguration(cc *config.KubeSchedulerConfiguration) utilerrors.Aggregate {
	var errs []error
	errs = append(errs, componentbasevalidation.ValidateClientConnectionConfiguration(&cc.ClientConnection, field.NewPath("clientConnection")).ToAggregate())
	errs = append(errs, componentbasevalidation.ValidateLeaderElectionConfiguration(&cc.LeaderElection, field.NewPath("leaderElection")).ToAggregate())
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
	for _, msg := range validation.IsValidSocketAddr(cc.HealthzBindAddress) {
		errs = append(errs, field.Invalid(field.NewPath("healthzBindAddress"), cc.HealthzBindAddress, msg))
	}
	for _, msg := range validation.IsValidSocketAddr(cc.MetricsBindAddress) {
		errs = append(errs, field.Invalid(field.NewPath("metricsBindAddress"), cc.MetricsBindAddress, msg))
	}
	if cc.PercentageOfNodesToScore < 0 || cc.PercentageOfNodesToScore > 100 {
		errs = append(errs, field.Invalid(field.NewPath("percentageOfNodesToScore"),
			cc.PercentageOfNodesToScore, "not in valid range [0-100]"))
	}
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

type removedPlugins struct {
	schemeGroupVersion string
	plugins            []string
}

// removedPluginsByVersion maintains a list of removed plugins in each version.
// Remember to add an entry to that list when creating a new component config
// version (even if the list of removed plugins is empty).
var removedPluginsByVersion = []removedPlugins{
	{
		schemeGroupVersion: v1beta1.SchemeGroupVersion.String(),
		plugins:            []string{},
	},
	{
		schemeGroupVersion: v1beta2.SchemeGroupVersion.String(),
		plugins: []string{
			"NodeLabel",
			"ServiceAffinity",
			"NodePreferAvoidPods",
			"NodeResourcesLeastAllocated",
			"NodeResourcesMostAllocated",
			"RequestedToCapacityRatio",
		},
	},
}

// conflictScorePluginsByVersion maintains a map of conflict plugins in each version.
// Remember to add an entry to that list when creating a new component config
// version (even if the list of conflict plugins is empty).
var conflictScorePluginsByVersion = map[string]map[string]sets.String{
	v1beta1.SchemeGroupVersion.String(): {
		"NodeResourcesFit": sets.NewString(
			"NodeResourcesLeastAllocated",
			"NodeResourcesMostAllocated",
			"RequestedToCapacityRatio"),
	},
	v1beta2.SchemeGroupVersion.String(): nil,
}

// isScorePluginConflict checks if a given plugin was conflict with other plugin in the given component
// config version or earlier.
func isScorePluginConflict(apiVersion string, name string, profile *config.KubeSchedulerProfile) []string {
	var conflictPlugins []string
	cp, ok := conflictScorePluginsByVersion[apiVersion]
	if !ok {
		return nil
	}
	plugin, ok := cp[name]
	if !ok {
		return nil
	}
	for _, p := range profile.Plugins.Score.Enabled {
		if plugin.Has(p.Name) {
			conflictPlugins = append(conflictPlugins, p.Name)
		}
	}
	return conflictPlugins
}

// isPluginRemoved checks if a given plugin was removed in the given component
// config version or earlier.
func isPluginRemoved(apiVersion string, name string) (bool, string) {
	for _, dp := range removedPluginsByVersion {
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

func validatePluginSetForRemovedPlugins(path *field.Path, apiVersion string, ps config.PluginSet) []error {
	var errs []error
	for i, plugin := range ps.Enabled {
		if removed, removedVersion := isPluginRemoved(apiVersion, plugin.Name); removed {
			errs = append(errs, field.Invalid(path.Child("enabled").Index(i), plugin.Name, fmt.Sprintf("was removed in version %q (KubeSchedulerConfiguration is version %q)", removedVersion, apiVersion)))
		}
	}
	return errs
}

func validateScorePluginSetForConflictPlugins(path *field.Path, apiVersion string, profile *config.KubeSchedulerProfile) []error {
	var errs []error
	for i, plugin := range profile.Plugins.Score.Enabled {
		if cp := isScorePluginConflict(apiVersion, plugin.Name, profile); len(cp) > 0 {
			errs = append(errs, field.Invalid(path.Child("enabled").Index(i), plugin.Name, fmt.Sprintf("was conflict with %q in version %q (KubeSchedulerConfiguration is version %q)", cp, apiVersion, apiVersion)))
		}
	}
	return errs
}

func validateKubeSchedulerProfile(path *field.Path, apiVersion string, profile *config.KubeSchedulerProfile) []error {
	var errs []error
	if len(profile.SchedulerName) == 0 {
		errs = append(errs, field.Required(path.Child("schedulerName"), ""))
	}
	errs = append(errs, validatePluginConfig(path, apiVersion, profile)...)
	return errs
}

func validatePluginConfig(path *field.Path, apiVersion string, profile *config.KubeSchedulerProfile) []error {
	var errs []error
	m := map[string]interface{}{
		"DefaultPreemption":               ValidateDefaultPreemptionArgs,
		"InterPodAffinity":                ValidateInterPodAffinityArgs,
		"NodeAffinity":                    ValidateNodeAffinityArgs,
		"NodeLabel":                       ValidateNodeLabelArgs,
		"NodeResourcesBalancedAllocation": ValidateNodeResourcesBalancedAllocationArgs,
		"NodeResourcesFitArgs":            ValidateNodeResourcesFitArgs,
		"NodeResourcesLeastAllocated":     ValidateNodeResourcesLeastAllocatedArgs,
		"NodeResourcesMostAllocated":      ValidateNodeResourcesMostAllocatedArgs,
		"PodTopologySpread":               ValidatePodTopologySpreadArgs,
		"RequestedToCapacityRatio":        ValidateRequestedToCapacityRatioArgs,
		"VolumeBinding":                   ValidateVolumeBindingArgs,
	}

	if profile.Plugins != nil {
		stagesToPluginSet := map[string]config.PluginSet{
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
			errs = append(errs, validatePluginSetForRemovedPlugins(
				pluginsPath.Child(s), apiVersion, p)...)
		}
		errs = append(errs, validateScorePluginSetForConflictPlugins(
			pluginsPath.Child("score"), apiVersion, profile)...)
	}

	seenPluginConfig := make(sets.String)

	for i := range profile.PluginConfig {
		pluginConfigPath := path.Child("pluginConfig").Index(i)
		name := profile.PluginConfig[i].Name
		args := profile.PluginConfig[i].Args
		if seenPluginConfig.Has(name) {
			errs = append(errs, field.Duplicate(pluginConfigPath, name))
		} else {
			seenPluginConfig.Insert(name)
		}
		if removed, removedVersion := isPluginRemoved(apiVersion, name); removed {
			errs = append(errs, field.Invalid(pluginConfigPath, name, fmt.Sprintf("was removed in version %q (KubeSchedulerConfiguration is version %q)", removedVersion, apiVersion)))
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
		if !cmp.Equal(canon, curr) {
			errs = append(errs, field.Invalid(path.Index(i).Child("plugins", "queueSort"), curr, "has to match for all profiles"))
		}
		for _, cfg := range profiles[i].PluginConfig {
			if cfg.Name == queueSortName && !cmp.Equal(queueSortArgs, cfg.Args) {
				errs = append(errs, field.Invalid(path.Index(i).Child("pluginConfig", "args"), cfg.Args, "has to match for all profiles"))
			}
		}
	}
	return errs
}

// ValidatePolicy checks for errors in the Config
// It does not return early so that it can find as many errors as possible
func ValidatePolicy(policy config.Policy) error {
	var validationErrors []error

	priorities := make(map[string]config.PriorityPolicy, len(policy.Priorities))
	for _, priority := range policy.Priorities {
		if priority.Weight <= 0 || priority.Weight >= config.MaxWeight {
			validationErrors = append(validationErrors, fmt.Errorf("priority %s should have a positive weight applied to it or it has overflown", priority.Name))
		}
		validationErrors = append(validationErrors, validateCustomPriorities(priorities, priority))
	}

	if extenderErrs := validateExtenders(field.NewPath("extenders"), policy.Extenders); len(extenderErrs) > 0 {
		validationErrors = append(validationErrors, extenderErrs...)
	}

	if policy.HardPodAffinitySymmetricWeight < 0 || policy.HardPodAffinitySymmetricWeight > 100 {
		validationErrors = append(validationErrors, field.Invalid(field.NewPath("hardPodAffinitySymmetricWeight"), policy.HardPodAffinitySymmetricWeight, "not in valid range [0-100]"))
	}
	return utilerrors.NewAggregate(validationErrors)
}

// validateExtenders validates the configured extenders for the Scheduler
func validateExtenders(fldPath *field.Path, extenders []config.Extender) []error {
	var errs []error
	binders := 0
	extenderManagedResources := sets.NewString()
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

// validateCustomPriorities validates that:
// 1. RequestedToCapacityRatioRedeclared custom priority cannot be declared multiple times,
// 2. LabelPreference/ServiceAntiAffinity custom priorities can be declared multiple times,
// however the weights for each custom priority type should be the same.
func validateCustomPriorities(priorities map[string]config.PriorityPolicy, priority config.PriorityPolicy) error {
	verifyRedeclaration := func(priorityType string) error {
		if existing, alreadyDeclared := priorities[priorityType]; alreadyDeclared {
			return fmt.Errorf("priority %q redeclares custom priority %q, from: %q", priority.Name, priorityType, existing.Name)
		}
		priorities[priorityType] = priority
		return nil
	}
	verifyDifferentWeights := func(priorityType string) error {
		if existing, alreadyDeclared := priorities[priorityType]; alreadyDeclared {
			if existing.Weight != priority.Weight {
				return fmt.Errorf("%s  priority %q has a different weight with %q", priorityType, priority.Name, existing.Name)
			}
		}
		priorities[priorityType] = priority
		return nil
	}
	if priority.Argument != nil {
		if priority.Argument.LabelPreference != nil {
			if err := verifyDifferentWeights("LabelPreference"); err != nil {
				return err
			}
		} else if priority.Argument.ServiceAntiAffinity != nil {
			if err := verifyDifferentWeights("ServiceAntiAffinity"); err != nil {
				return err
			}
		} else if priority.Argument.RequestedToCapacityRatioArguments != nil {
			if err := verifyRedeclaration("RequestedToCapacityRatio"); err != nil {
				return err
			}
		} else {
			return fmt.Errorf("no priority arguments set for priority %s", priority.Name)
		}
	}
	return nil
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
