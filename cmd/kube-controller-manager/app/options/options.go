/*
Copyright 2014 The Kubernetes Authors.

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

// Package options provides the flags used for the controller manager.
//
package options

import (
	"fmt"
	"strings"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cmoptions "k8s.io/kubernetes/cmd/controller-manager/app/options"
	kubecontrollerconfig "k8s.io/kubernetes/cmd/kube-controller-manager/app/config"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	"k8s.io/kubernetes/pkg/master/ports"

	// add the kubernetes feature gates
	_ "k8s.io/kubernetes/pkg/features"

	"github.com/spf13/pflag"
)

// KubeControllerManagerOptions is the main context object for the controller manager.
type KubeControllerManagerOptions struct {
	Generic *cmoptions.GenericControllerManagerOptions
}

// NewKubeControllerManagerOptions creates a new KubeControllerManagerOptions with a default config.
func NewKubeControllerManagerOptions() *KubeControllerManagerOptions {
	componentConfig := cmoptions.NewDefaultControllerManagerComponentConfig(ports.InsecureKubeControllerManagerPort)
	s := KubeControllerManagerOptions{
		// The common/default are kept in 'cmd/kube-controller-manager/app/options/util.go'.
		// Please make common changes there but put anything kube-controller specific here.
		Generic: cmoptions.NewGenericControllerManagerOptions(componentConfig),
	}

	s.Generic.SecureServing.ServerCert.CertDirectory = "/var/run/kubernetes"
	s.Generic.SecureServing.ServerCert.PairName = "kube-controller-manager"

	gcIgnoredResources := make([]componentconfig.GroupResource, 0, len(garbagecollector.DefaultIgnoredResources()))
	for r := range garbagecollector.DefaultIgnoredResources() {
		gcIgnoredResources = append(gcIgnoredResources, componentconfig.GroupResource{Group: r.Group, Resource: r.Resource})
	}

	s.Generic.GarbageCollectorController.GCIgnoredResources = gcIgnoredResources

	return &s
}

// AddFlags adds flags for a specific KubeControllerManagerOptions to the specified FlagSet
func (s *KubeControllerManagerOptions) AddFlags(fs *pflag.FlagSet, allControllers []string, disabledByDefaultControllers []string) {
	s.Generic.AddFlags(fs)
	s.Generic.AttachDetachController.AddFlags(fs)
	s.Generic.CSRSigningController.AddFlags(fs)
	s.Generic.DeploymentController.AddFlags(fs)
	s.Generic.DaemonSetController.AddFlags(fs)
	s.Generic.DeprecatedFlags.AddFlags(fs)
	s.Generic.EndPointController.AddFlags(fs)
	s.Generic.GarbageCollectorController.AddFlags(fs)
	s.Generic.HPAController.AddFlags(fs)
	s.Generic.JobController.AddFlags(fs)
	s.Generic.NamespaceController.AddFlags(fs)
	s.Generic.NodeIpamController.AddFlags(fs)
	s.Generic.NodeLifecycleController.AddFlags(fs)
	s.Generic.PersistentVolumeBinderController.AddFlags(fs)
	s.Generic.PodGCController.AddFlags(fs)
	s.Generic.ReplicaSetController.AddFlags(fs)
	s.Generic.ReplicationController.AddFlags(fs)
	s.Generic.ResourceQuotaController.AddFlags(fs)
	s.Generic.SAController.AddFlags(fs)

	fs.StringSliceVar(&s.Generic.Controllers, "controllers", s.Generic.Controllers, fmt.Sprintf(""+
		"A list of controllers to enable.  '*' enables all on-by-default controllers, 'foo' enables the controller "+
		"named 'foo', '-foo' disables the controller named 'foo'.\nAll controllers: %s\nDisabled-by-default controllers: %s",
		strings.Join(allControllers, ", "), strings.Join(disabledByDefaultControllers, ", ")))
	fs.StringVar(&s.Generic.ExternalCloudVolumePlugin, "external-cloud-volume-plugin", s.Generic.ExternalCloudVolumePlugin, "The plugin to use when cloud provider is set to external. Can be empty, should only be set when cloud-provider is external. Currently used to allow node and volume controllers to work for in tree cloud providers.")
	var dummy string
	fs.MarkDeprecated("insecure-experimental-approve-all-kubelet-csrs-for-group", "This flag does nothing.")
	fs.StringVar(&dummy, "insecure-experimental-approve-all-kubelet-csrs-for-group", "", "This flag does nothing.")
	utilfeature.DefaultFeatureGate.AddFlag(fs)
}

// ApplyTo fills up controller manager config with options.
func (s *KubeControllerManagerOptions) ApplyTo(c *kubecontrollerconfig.Config) error {
	err := s.Generic.ApplyTo(&c.Generic, "controller-manager")

	c.Generic.ComponentConfig.Controllers = s.Generic.Controllers
	c.Generic.ComponentConfig.ExternalCloudVolumePlugin = s.Generic.ExternalCloudVolumePlugin

	return err
}

// Validate is used to validate the options and config before launching the controller manager
func (s *KubeControllerManagerOptions) Validate(allControllers []string, disabledByDefaultControllers []string) error {
	var errs []error

	allControllersSet := sets.NewString(allControllers...)
	for _, controller := range s.Generic.Controllers {
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

	return utilerrors.NewAggregate(errs)
}

// Config return a controller manager config objective
func (s KubeControllerManagerOptions) Config(allControllers []string, disabledByDefaultControllers []string) (*kubecontrollerconfig.Config, error) {
	if err := s.Validate(allControllers, disabledByDefaultControllers); err != nil {
		return nil, err
	}

	c := &kubecontrollerconfig.Config{}
	if err := s.ApplyTo(c); err != nil {
		return nil, err
	}

	return c, nil
}
