/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

// Package webhook implements a generic HTTP webhook plugin.
package webhook

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/runtime"
	runtimeserializer "k8s.io/kubernetes/pkg/runtime/serializer"

	_ "k8s.io/kubernetes/pkg/apis/authorization/install"
)

type GenericWebhook struct {
	RestClient *restclient.RESTClient
}

// New creates a new GenericWebhook from the provided kubeconfig file.
func NewGenericWebhook(kubeConfigFile string, groupVersions []unversioned.GroupVersion) (*GenericWebhook, error) {
	for _, groupVersion := range groupVersions {
		if !registered.IsEnabledVersion(groupVersion) {
			return nil, fmt.Errorf("webhook plugin requires enabling extension resource: %s", groupVersion)
		}
	}

	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	loadingRules.ExplicitPath = kubeConfigFile
	loader := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{})

	clientConfig, err := loader.ClientConfig()
	if err != nil {
		return nil, err
	}
	codec := api.Codecs.LegacyCodec(groupVersions...)
	clientConfig.ContentConfig.NegotiatedSerializer = runtimeserializer.NegotiatedSerializerWrapper(
		runtime.SerializerInfo{Serializer: codec},
		runtime.StreamSerializerInfo{},
	)

	restClient, err := restclient.UnversionedRESTClientFor(clientConfig)
	if err != nil {
		return nil, err
	}

	// TODO(ericchiang): Can we ensure remote service is reachable?

	return &GenericWebhook{restClient}, nil
}
