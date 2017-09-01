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

	"github.com/golang/glog"

	"k8s.io/api/admissionregistration/v1alpha1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

type ExternalAdmissionHookConfigurationLister interface {
	List(opts metav1.ListOptions) (*v1alpha1.ExternalAdmissionHookConfigurationList, error)
}

type ExternalAdmissionHookConfigurationManager struct {
	*poller
}

func NewExternalAdmissionHookConfigurationManager(c ExternalAdmissionHookConfigurationLister) *ExternalAdmissionHookConfigurationManager {
	getFn := func() (runtime.Object, error) {
		list, err := c.List(metav1.ListOptions{})
		if err != nil {
			if errors.IsNotFound(err) || errors.IsForbidden(err) {
				glog.V(5).Infof("ExternalAdmissionHookConfiguration are disabled due to an error: %v", err)
				return nil, ErrDisabled
			}
			return nil, err
		}
		return mergeExternalAdmissionHookConfigurations(list), nil
	}

	return &ExternalAdmissionHookConfigurationManager{
		newPoller(getFn),
	}
}

// ExternalAdmissionHooks returns the merged ExternalAdmissionHookConfiguration.
func (im *ExternalAdmissionHookConfigurationManager) ExternalAdmissionHooks() (*v1alpha1.ExternalAdmissionHookConfiguration, error) {
	configuration, err := im.poller.configuration()
	if err != nil {
		return nil, err
	}
	externalAdmissionHookConfiguration, ok := configuration.(*v1alpha1.ExternalAdmissionHookConfiguration)
	if !ok {
		return nil, fmt.Errorf("expected type %v, got type %v", reflect.TypeOf(externalAdmissionHookConfiguration), reflect.TypeOf(configuration))
	}
	return externalAdmissionHookConfiguration, nil
}

func (im *ExternalAdmissionHookConfigurationManager) Run(stopCh <-chan struct{}) {
	im.poller.Run(stopCh)
}

func mergeExternalAdmissionHookConfigurations(
	list *v1alpha1.ExternalAdmissionHookConfigurationList,
) *v1alpha1.ExternalAdmissionHookConfiguration {
	configurations := list.Items
	var ret v1alpha1.ExternalAdmissionHookConfiguration
	for _, c := range configurations {
		ret.ExternalAdmissionHooks = append(ret.ExternalAdmissionHooks, c.ExternalAdmissionHooks...)
	}
	return &ret
}
