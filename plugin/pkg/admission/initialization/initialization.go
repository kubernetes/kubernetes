/*
Copyright 2017 The Kubernetes Authors.

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

package initialization

import (
	"fmt"
	"io"
	"strings"

	"github.com/golang/glog"

	"k8s.io/api/admissionregistration/v1alpha1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubeapiserver/admission/configuration"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register("Initializers", func(config io.Reader) (admission.Interface, error) {
		return NewInitializer(), nil
	})
}

type initializerOptions struct {
	Initializers []string
}

type InitializationConfig interface {
	Run(stopCh <-chan struct{})
	Initializers() (*v1alpha1.InitializerConfiguration, error)
}

type initializer struct {
	config     InitializationConfig
	authorizer authorizer.Authorizer
}

// NewInitializer creates a new initializer plugin which assigns newly created resources initializers
// based on configuration loaded from the admission API group.
// FUTURE: this may be moved to the storage layer of the apiserver, but for now this is an alpha feature
//   that can be disabled.
func NewInitializer() admission.Interface {
	return &initializer{}
}

func (i *initializer) Validate() error {
	if i.config == nil {
		return fmt.Errorf("the Initializer admission plugin requires a Kubernetes client to be provided")
	}
	if i.authorizer == nil {
		return fmt.Errorf("the Initializer admission plugin requires an authorizer to be provided")
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.Initializers) {
		if err := utilfeature.DefaultFeatureGate.Set(string(features.Initializers) + "=true"); err != nil {
			glog.Errorf("error enabling Initializers feature as part of admission plugin setup: %v", err)
		} else {
			glog.Infof("enabled Initializers feature as part of admission plugin setup")
		}
	}

	i.config.Run(wait.NeverStop)
	return nil
}

func (i *initializer) SetExternalKubeClientSet(client clientset.Interface) {
	i.config = configuration.NewInitializerConfigurationManager(client.Admissionregistration().InitializerConfigurations())
}

func (i *initializer) SetAuthorizer(a authorizer.Authorizer) {
	i.authorizer = a
}

var initializerFieldPath = field.NewPath("metadata", "initializers")

// readConfig holds requests instead of failing them if the server is not yet initialized
// or is unresponsive. It formats the returned error for client use if necessary.
func (i *initializer) readConfig(a admission.Attributes) (*v1alpha1.InitializerConfiguration, error) {
	// read initializers from config
	config, err := i.config.Initializers()
	if err == nil {
		return config, nil
	}

	// if initializer configuration is disabled, fail open
	if err == configuration.ErrDisabled {
		return &v1alpha1.InitializerConfiguration{}, nil
	}

	e := errors.NewServerTimeout(a.GetResource().GroupResource(), "create", 1)
	if err == configuration.ErrNotReady {
		e.ErrStatus.Message = fmt.Sprintf("Waiting for initialization configuration to load: %v", err)
		e.ErrStatus.Reason = "LoadingConfiguration"
		e.ErrStatus.Details.Causes = append(e.ErrStatus.Details.Causes, metav1.StatusCause{
			Type:    "InitializerConfigurationPending",
			Message: "The server is waiting for the initializer configuration to be loaded.",
		})
	} else {
		e.ErrStatus.Message = fmt.Sprintf("Unable to refresh the initializer configuration: %v", err)
		e.ErrStatus.Reason = "LoadingConfiguration"
		e.ErrStatus.Details.Causes = append(e.ErrStatus.Details.Causes, metav1.StatusCause{
			Type:    "InitializerConfigurationFailure",
			Message: "An error has occurred while refreshing the initializer configuration, no resources can be created until a refresh succeeds.",
		})
	}
	return nil, e
}

// Admit checks for create requests to add initializers, or update request to enforce invariants.
// The admission controller fails open if the object doesn't have ObjectMeta (can't be initialized).
// A client with sufficient permission ("initialize" verb on resource) can specify its own initializers
// or an empty initializers struct (which bypasses initialization). Only clients with the initialize verb
// can update objects that have not completed initialization. Sub resources can still be modified on
// resources that are undergoing initialization.
// TODO: once this logic is ready for beta, move it into the REST storage layer.
func (i *initializer) Admit(a admission.Attributes) (err error) {
	switch a.GetOperation() {
	case admission.Create, admission.Update:
	default:
		return nil
	}

	// TODO: should sub-resource action should be denied until the object is initialized?
	if len(a.GetSubresource()) > 0 {
		return nil
	}

	switch a.GetOperation() {
	case admission.Create:
		accessor, err := meta.Accessor(a.GetObject())
		if err != nil {
			// objects without meta accessor cannot be checked for initialization, and it is possible to make calls
			// via our API that don't have ObjectMeta
			return nil
		}
		existing := accessor.GetInitializers()
		if existing != nil {
			glog.V(5).Infof("Admin bypassing initialization for %s", a.GetResource())

			// it must be possible for some users to bypass initialization - for now, check the initialize operation
			if err := i.canInitialize(a, "create with initializers denied"); err != nil {
				return err
			}
			// allow administrators to bypass initialization by setting an empty initializers struct
			if len(existing.Pending) == 0 && existing.Result == nil {
				accessor.SetInitializers(nil)
				return nil
			}
		} else {
			glog.V(5).Infof("Checking initialization for %s", a.GetResource())

			config, err := i.readConfig(a)
			if err != nil {
				return err
			}

			// Mirror pods are exempt from initialization because they are created and initialized
			// on the Kubelet before they appear in the API.
			// TODO: once this moves to REST storage layer, this becomes a pod specific concern
			if pod, ok := a.GetObject().(*api.Pod); ok && pod != nil {
				if _, isMirror := pod.Annotations[api.MirrorPodAnnotationKey]; isMirror {
					return nil
				}
			}

			names := findInitializers(config, a.GetResource())
			if len(names) == 0 {
				glog.V(5).Infof("No initializers needed")
				return nil
			}

			glog.V(5).Infof("Found initializers for %s: %v", a.GetResource(), names)
			accessor.SetInitializers(newInitializers(names))
		}

	case admission.Update:
		accessor, err := meta.Accessor(a.GetObject())
		if err != nil {
			// objects without meta accessor cannot be checked for initialization, and it is possible to make calls
			// via our API that don't have ObjectMeta
			return nil
		}
		updated := accessor.GetInitializers()

		existingAccessor, err := meta.Accessor(a.GetOldObject())
		if err != nil {
			// if the old object does not have an accessor, but the new one does, error out
			return fmt.Errorf("initialized resources must be able to set initializers (%T): %v", a.GetOldObject(), err)
		}
		existing := existingAccessor.GetInitializers()

		// updates on initialized resources are allowed
		if updated == nil && existing == nil {
			return nil
		}

		glog.V(5).Infof("Modifying uninitialized resource %s", a.GetResource())

		if updated != nil && len(updated.Pending) == 0 && updated.Result == nil {
			accessor.SetInitializers(nil)
		}
		// because we are called before validation, we need to ensure the update transition is valid.
		if errs := validation.ValidateInitializersUpdate(updated, existing, initializerFieldPath); len(errs) > 0 {
			return errors.NewInvalid(a.GetKind().GroupKind(), a.GetName(), errs)
		}

		// caller must have the ability to mutate un-initialized resources
		if err := i.canInitialize(a, "update to uninitialized resource denied"); err != nil {
			return err
		}

		// TODO: restrict initialization list changes to specific clients?
	}

	return nil
}

func (i *initializer) canInitialize(a admission.Attributes, message string) error {
	// caller must have the ability to mutate un-initialized resources
	authorized, reason, err := i.authorizer.Authorize(authorizer.AttributesRecord{
		Name:            a.GetName(),
		ResourceRequest: true,
		User:            a.GetUserInfo(),
		Verb:            "initialize",
		Namespace:       a.GetNamespace(),
		APIGroup:        a.GetResource().Group,
		APIVersion:      a.GetResource().Version,
		Resource:        a.GetResource().Resource,
	})
	if err != nil {
		return err
	}
	if !authorized {
		return errors.NewForbidden(a.GetResource().GroupResource(), a.GetName(), fmt.Errorf("%s: %s", message, reason))
	}
	return nil
}

func (i *initializer) Handles(op admission.Operation) bool {
	return op == admission.Create || op == admission.Update
}

// newInitializers populates an Initializers struct.
func newInitializers(names []string) *metav1.Initializers {
	if len(names) == 0 {
		return nil
	}
	var init []metav1.Initializer
	for _, name := range names {
		init = append(init, metav1.Initializer{Name: name})
	}
	return &metav1.Initializers{
		Pending: init,
	}
}

// findInitializers returns the list of initializer names that apply to a config. It returns an empty list
// if no initializers apply.
func findInitializers(initializers *v1alpha1.InitializerConfiguration, gvr schema.GroupVersionResource) []string {
	var names []string
	for _, init := range initializers.Initializers {
		if !matchRule(init.Rules, gvr) {
			continue
		}
		names = append(names, init.Name)
	}
	return names
}

// matchRule returns true if any rule matches the provided group version resource.
func matchRule(rules []v1alpha1.Rule, gvr schema.GroupVersionResource) bool {
	for _, rule := range rules {
		if !hasGroup(rule.APIGroups, gvr.Group) {
			return false
		}
		if !hasVersion(rule.APIVersions, gvr.Version) {
			return false
		}
		if !hasResource(rule.Resources, gvr.Resource) {
			return false
		}
	}
	return len(rules) > 0
}

func hasGroup(groups []string, group string) bool {
	if groups[0] == "*" {
		return true
	}
	for _, g := range groups {
		if g == group {
			return true
		}
	}
	return false
}

func hasVersion(versions []string, version string) bool {
	if versions[0] == "*" {
		return true
	}
	for _, v := range versions {
		if v == version {
			return true
		}
	}
	return false
}

func hasResource(resources []string, resource string) bool {
	if resources[0] == "*" || resources[0] == "*/*" {
		return true
	}
	for _, r := range resources {
		if strings.Contains(r, "/") {
			continue
		}
		if r == resource {
			return true
		}
	}
	return false
}
