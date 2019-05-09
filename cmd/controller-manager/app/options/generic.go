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
	componentbaseconfig "k8s.io/component-base/config"
	"k8s.io/kubernetes/pkg/client/leaderelectionconfig"
	kubectrlmgrconfig "k8s.io/kubernetes/pkg/controller/apis/config"
)

// GenericControllerManagerConfigurationOptions holds the options which are generic.
type GenericControllerManagerConfigurationOptions struct {
	*kubectrlmgrconfig.GenericControllerManagerConfiguration
	Debugging *DebuggingOptions
}

// NewGenericControllerManagerConfigurationOptions returns generic configuration default values for both
// the kube-controller-manager and the cloud-contoller-manager. Any common changes should
// be made here. Any individual changes should be made in that controller.
func NewGenericControllerManagerConfigurationOptions(cfg *kubectrlmgrconfig.GenericControllerManagerConfiguration) *GenericControllerManagerConfigurationOptions {
	o := &GenericControllerManagerConfigurationOptions{
		GenericControllerManagerConfiguration: cfg,
		Debugging: &DebuggingOptions{
			DebuggingConfiguration: &componentbaseconfig.DebuggingConfiguration{},
		},
	}

	return o
}

// AddFlags adds flags related to generic for controller manager to the specified FlagSet.
func (o *GenericControllerManagerConfigurationOptions) AddFlags(fss *cliflag.NamedFlagSets, allControllers, disabledByDefaultControllers []string) {
	if o == nil {
		return
	}

	o.Debugging.AddFlags(fss.FlagSet("debugging"))
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

	leaderelectionconfig.BindFlags(&o.LeaderElection, genericfs)
}

// ApplyTo fills up generic config with options.
func (o *GenericControllerManagerConfigurationOptions) ApplyTo(cfg *kubectrlmgrconfig.GenericControllerManagerConfiguration) error {
	if o == nil {
		return nil
	}

	if err := o.Debugging.ApplyTo(&cfg.Debugging); err != nil {
		return err
	}

	cfg.Port = o.Port
	cfg.Address = o.Address
	cfg.MinResyncPeriod = o.MinResyncPeriod
	cfg.ClientConnection = o.ClientConnection
	cfg.ControllerStartInterval = o.ControllerStartInterval
	cfg.LeaderElection = o.LeaderElection
	cfg.Controllers = o.Controllers

	return nil
}

// Validate checks validation of GenericOptions.
func (o *GenericControllerManagerConfigurationOptions) Validate(allControllers []string, disabledByDefaultControllers []string) []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	errs = append(errs, o.Debugging.Validate()...)

	allControllersSet := sets.NewString(allControllers...)
	for _, controller := range o.Controllers {
		if controller == "*" {
			continue
		}
		if strings.HasPrefix(controller, "-") {
			controller = controller[1:]
		}
		if !allControllersSet.Has(controller) {
			errs = append(errs, fmt.Errorf("%q is not in the list of known controllers", controller))
		}
	}

	return errs
}
