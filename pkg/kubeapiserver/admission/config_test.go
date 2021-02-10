/*
Copyright 2021 The Kubernetes Authors.

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

package admission

import (
	"net/http"
	"net/url"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/admission"
	webhookinit "k8s.io/apiserver/pkg/admission/plugin/webhook/initializer"
	egressselector "k8s.io/apiserver/pkg/server/egressselector"
	"k8s.io/apiserver/pkg/util/webhook"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	externalinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	quotainstall "k8s.io/kubernetes/pkg/quota/v1/install"
)

type fakeServiceResolver struct{}

func (f *fakeServiceResolver) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	return nil, nil
}

func TestConfigNew(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	sharedInformers := externalinformers.NewSharedInformerFactory(fakeClient, 10*time.Minute)

	acceptableConfig := &Config{
		ExternalInformers:    sharedInformers,
		LoopbackClientConfig: &rest.Config{},
		CloudConfigFile:      "",
	}

	unacceptableConfig := &Config{
		ExternalInformers:    sharedInformers,
		LoopbackClientConfig: &rest.Config{RateLimiter: nil, QPS: 1, Burst: -1},
		CloudConfigFile:      "",
	}

	transport := utilnet.SetTransportDefaults(&http.Transport{})
	egressSelector := &egressselector.EgressSelector{}
	serviceResolver := &fakeServiceResolver{}

	webhookAuthResolverWrapper := webhook.NewDefaultAuthenticationInfoResolverWrapper(transport, egressSelector, acceptableConfig.LoopbackClientConfig)
	webhookPluginInitializer := webhookinit.NewPluginInitializer(webhookAuthResolverWrapper, serviceResolver)

	clientset, _ := kubernetes.NewForConfig(acceptableConfig.LoopbackClientConfig)
	discoveryClient := cacheddiscovery.NewMemCacheClient(clientset.Discovery())
	discoveryRESTMapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryClient)

	var cloudConfig []byte
	kubePluginInitializer := NewPluginInitializer(
		cloudConfig,
		discoveryRESTMapper,
		quotainstall.NewQuotaConfigurationForAdmission(),
	)

	testcases := []struct {
		description                string
		config                     *Config
		exceptedPluginInitializers []admission.PluginInitializer
		exceptedErr                bool
	}{{
		description:                "Acceptable config should return admission and webhook PluginInitializer.",
		config:                     acceptableConfig,
		exceptedPluginInitializers: []admission.PluginInitializer{webhookPluginInitializer, kubePluginInitializer},
		exceptedErr:                false,
	}, {
		description:                "Unacceptable config should return error",
		config:                     unacceptableConfig,
		exceptedPluginInitializers: nil,
		exceptedErr:                true,
	}}

	cmpOption := cmp.Options{
		cmp.AllowUnexported(webhookinit.PluginInitializer{}),
		// authenticationInfoResolverWrapper ignores because the function pointer will set and the comparison fails.
		cmpopts.IgnoreFields(webhookinit.PluginInitializer{}, "authenticationInfoResolverWrapper"),
		cmp.AllowUnexported(PluginInitializer{}),
		// restMapper ignores because the order of data changes and the comparison fails.
		// quotaConfiguration ignores because it is an interface and cannot be compared.
		cmpopts.IgnoreFields(PluginInitializer{}, "restMapper", "quotaConfiguration"),
	}

	for _, tc := range testcases {

		t.Run(tc.description, func(t *testing.T) {
			pluginInitializers, _, err := tc.config.New(transport, egressSelector, serviceResolver)
			if diff := cmp.Diff(tc.exceptedPluginInitializers, pluginInitializers, cmpOption); diff != "" {
				t.Errorf("unexpected values  diff (-want, +got):\n%s", diff)
			}

			if tc.exceptedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
