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

package resourcequota

import (
	"fmt"
	"io"
	"io/ioutil"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	resourcequotaapi "k8s.io/kubernetes/plugin/pkg/admission/resourcequota/apis/resourcequota"
	"k8s.io/kubernetes/plugin/pkg/admission/resourcequota/apis/resourcequota/install"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	install.Install(scheme)
}

// AddCriticalPodLimitedResources limits the number of critical pods that can be created. We're creating a default
// resource quota in `kube-system` namespace to allow unlimited number of critical pods to be created in that
// namespace. Making this function public for easy testing
func AddCriticalPodLimitedResources() []resourcequotaapi.LimitedResource {
	return []resourcequotaapi.LimitedResource{
		{
			Resource: "pods",
			MatchScopes: []v1.ScopedResourceSelectorRequirement{
				{
					ScopeName: v1.ResourceQuotaScopePriorityClass,
					Operator:  v1.ScopeSelectorOpIn,
					Values:    []string{"system-cluster-critical", "system-node-critical"},
				},
			},
		},
	}
}

// LoadConfiguration loads the provided configuration.
func LoadConfiguration(config io.Reader) (*resourcequotaapi.Configuration, error) {
	// if no config is provided, return a default configuration with critical pods as limited resources
	if config == nil {
		config := &resourcequotaapi.Configuration{}
		scheme.Default(config)
		// append pods as limited resources so that we can limit the number of critical pods to be created. We have
		// a matching default quota in `kube-system` namespace which allows unlimited pods to be created in that
		// namespace
		config.LimitedResources = append(config.LimitedResources, AddCriticalPodLimitedResources()...)
		return config, nil
	}

	// we have a config so parse it and limit the critical pods that can be created
	data, err := ioutil.ReadAll(config)
	if err != nil {
		return nil, err
	}
	decoder := codecs.UniversalDecoder()
	decodedObj, err := runtime.Decode(decoder, data)
	if err != nil {
		return nil, err
	}
	resourceQuotaConfiguration, ok := decodedObj.(*resourcequotaapi.Configuration)
	if !ok {
		return nil, fmt.Errorf("unexpected type: %T", decodedObj)
	}
	// append pods as limited resources so that we can limit the number of critical pods to be created. We have
	// a matching default quota in `kube-system` namespace which allows unlimited pods to be created in that namespace
	resourceQuotaConfiguration.LimitedResources = append(resourceQuotaConfiguration.LimitedResources, AddCriticalPodLimitedResources()...)
	return resourceQuotaConfiguration, nil
}
