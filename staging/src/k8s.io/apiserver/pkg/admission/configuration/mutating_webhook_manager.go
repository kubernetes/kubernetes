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

package configuration

import (
	"fmt"
	"reflect"
	"sort"

	"github.com/golang/glog"

	"k8s.io/api/admissionregistration/v1alpha1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

type MutatingWebhookConfigurationLister interface {
	List(opts metav1.ListOptions) (*v1alpha1.MutatingWebhookConfigurationList, error)
}

// MutatingWebhookConfigurationManager collects the mutating webhook objects so that they can be called.
type MutatingWebhookConfigurationManager struct {
	*poller
}

func NewMutatingWebhookConfigurationManager(c MutatingWebhookConfigurationLister) *MutatingWebhookConfigurationManager {
	getFn := func() (runtime.Object, error) {
		list, err := c.List(metav1.ListOptions{})
		if err != nil {
			if errors.IsNotFound(err) || errors.IsForbidden(err) {
				glog.V(5).Infof("MutatingWebhookConfiguration are disabled due to an error: %v", err)
				return nil, ErrDisabled
			}
			return nil, err
		}
		return mergeMutatingWebhookConfigurations(list), nil
	}

	return &MutatingWebhookConfigurationManager{
		newPoller(getFn),
	}
}

// Webhooks returns the merged MutatingWebhookConfiguration.
func (im *MutatingWebhookConfigurationManager) Webhooks() (*v1alpha1.MutatingWebhookConfiguration, error) {
	configuration, err := im.poller.configuration()
	if err != nil {
		return nil, err
	}
	mutatingWebhookConfiguration, ok := configuration.(*v1alpha1.MutatingWebhookConfiguration)
	if !ok {
		return nil, fmt.Errorf("expected type %v, got type %v", reflect.TypeOf(mutatingWebhookConfiguration), reflect.TypeOf(configuration))
	}
	return mutatingWebhookConfiguration, nil
}

func (im *MutatingWebhookConfigurationManager) Run(stopCh <-chan struct{}) {
	im.poller.Run(stopCh)
}

func mergeMutatingWebhookConfigurations(
	list *v1alpha1.MutatingWebhookConfigurationList,
) *v1alpha1.MutatingWebhookConfiguration {
	configurations := append([]v1alpha1.MutatingWebhookConfiguration{}, list.Items...)
	var ret v1alpha1.MutatingWebhookConfiguration
	// The internal order of webhooks for each configuration is provided by the user
	// but configurations themselves can be in any order. As we are going to run these
	// webhooks in serial, they are sorted here to have a deterministic order.
	sort.Sort(byName(configurations))
	for _, c := range configurations {
		ret.Webhooks = append(ret.Webhooks, c.Webhooks...)
	}
	return &ret
}

// byName sorts MutatingWebhookConfiguration by name. These objects are all in
// cluster namespace (aka no namespace) thus they all have unique names.
type byName []v1alpha1.MutatingWebhookConfiguration

func (x byName) Len() int { return len(x) }

func (x byName) Swap(i, j int) { x[i], x[j] = x[j], x[i] }

func (x byName) Less(i, j int) bool {
	return x[i].ObjectMeta.Name < x[j].ObjectMeta.Name
}
