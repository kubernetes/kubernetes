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

package options

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/config/options"
	cmconfig "k8s.io/controller-manager/config"
	migration "k8s.io/controller-manager/pkg/leadermigration/options"
)

// GenericControllerManagerConfigurationOptions holds the options which are generic.
type GenericControllerManagerConfigurationOptions struct {
	*cmconfig.GenericControllerManagerConfiguration
	Debugging *DebuggingOptions
	// LeaderMigration is the options for leader migration, a nil indicates default options should be applied.
	LeaderMigration *migration.LeaderMigrationOptions
}

// NewGenericControllerManagerConfigurationOptions returns generic configuration default values for both
// the kube-controller-manager and the cloud-contoller-manager. Any common changes should
// be made here. Any individual changes should be made in that controller.
func NewGenericControllerManagerConfigurationOptions(cfg *cmconfig.GenericControllerManagerConfiguration) *GenericControllerManagerConfigurationOptions {
	o := &GenericControllerManagerConfigurationOptions{
		GenericControllerManagerConfiguration: cfg,
		Debugging:                             RecommendedDebuggingOptions(),
		LeaderMigration:                       &migration.LeaderMigrationOptions{},
	}

	return o
}

// AddFlags adds flags related to generic for controller manager to the specified FlagSet.
func (o *GenericControllerManagerConfigurationOptions) AddFlags(fss *cliflag.NamedFlagSets, allControllers, disabledByDefaultControllers []string, controllerAliases map[string]string) {
	if o == nil {
		return
	}

	o.Debugging.AddFlags(fss.FlagSet("debugging"))
	o.LeaderMigration.AddFlags(fss.FlagSet("leader-migration"))
	genericfs := fss.FlagSet("generic")
	genericfs.DurationVar(&o.MinResyncPeriod.Duration, "min-resync-period", o.MinResyncPeriod.Duration, "The resync period in reflectors will be random between MinResyncPeriod and 2*MinResyncPeriod.")
	genericfs.StringVar(&o.ClientConnection.ContentType, "kube-api-content-type", o.ClientConnection.ContentType, "Content type of requests sent to apiserver.")
	genericfs.Float32Var(&o.ClientConnection.QPS, "kube-api-qps", o.ClientConnection.QPS, "QPS to use while talking with kubernetes apiserver.")
	genericfs.Int32Var(&o.ClientConnection.Burst, "kube-api-burst", o.ClientConnection.Burst, "Burst to use while talking with kubernetes apiserver.")
	genericfs.DurationVar(&o.ControllerStartInterval.Duration, "controller-start-interval", o.ControllerStartInterval.Duration, "Interval between starting controller managers.")
	genericfs.StringSliceVar(&o.Controllers, "controllers", o.Controllers, fmt.Sprintf(""+
		"A list of controllers to enable. '*' enables all on-by-default controllers, 'foo' enables the controller "+
		"named 'foo', '-foo' disables the controller named 'foo'.\nAll controllers: %s\nDisabled-by-default controllers: %s",
		strings.Join(allControllers, ", "), strings.Join(disabledByDefaultControllers, ", ")))

	options.BindLeaderElectionFlags(&o.LeaderElection, genericfs)
}

// ApplyTo fills up generic config with options.
func (o *GenericControllerManagerConfigurationOptions) ApplyTo(cfg *cmconfig.GenericControllerManagerConfiguration, allControllers []string, disabledByDefaultControllers []string, controllerAliases map[string]string) error {
	if o == nil {
		return nil
	}

	if err := o.Debugging.ApplyTo(&cfg.Debugging); err != nil {
		return err
	}
	if err := o.LeaderMigration.ApplyTo(cfg); err != nil {
		return err
	}
	cfg.Port = o.Port
	cfg.Address = o.Address
	cfg.MinResyncPeriod = o.MinResyncPeriod
	cfg.ClientConnection = o.ClientConnection
	cfg.ControllerStartInterval = o.ControllerStartInterval
	cfg.LeaderElection = o.LeaderElection

	// copy controller names and replace aliases with canonical names
	cfg.Controllers = make([]string, len(o.Controllers))
	for i, initialName := range o.Controllers {
		initialNameWithoutPrefix := strings.TrimPrefix(initialName, "-")
		controllerName := initialNameWithoutPrefix
		if canonicalName, ok := controllerAliases[controllerName]; ok {
			controllerName = canonicalName
		}
		if strings.HasPrefix(initialName, "-") {
			controllerName = fmt.Sprintf("-%s", controllerName)
		}
		cfg.Controllers[i] = controllerName
	}

	return nil
}

// Validate checks validation of GenericOptions.
func (o *GenericControllerManagerConfigurationOptions) Validate(allControllers []string, disabledByDefaultControllers []string, controllerAliases map[string]string) []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	errs = append(errs, o.Debugging.Validate()...)

	// TODO: This can be removed when ResourceLock is not available
	// Lock the ResourceLock using leases
	if o.LeaderElection.LeaderElect && o.LeaderElection.ResourceLock != "leases" {
		errs = append(errs, fmt.Errorf(`resourceLock value must be "leases"`))
	}

	allControllersSet := sets.NewString(allControllers...)
	for _, initialName := range o.Controllers {
		if initialName == "*" {
			continue
		}
		initialNameWithoutPrefix := strings.TrimPrefix(initialName, "-")
		controllerName := initialNameWithoutPrefix
		if canonicalName, ok := controllerAliases[controllerName]; ok {
			controllerName = canonicalName
		}
		if !allControllersSet.Has(controllerName) {
			errs = append(errs, fmt.Errorf("%q is not in the list of known controllers", initialNameWithoutPrefix))
		}
	}

	return errs
}
