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

package clientcmd

import (
	"fmt"
	"testing"

	"k8s.io/client-go/1.5/rest"
	clientcmdapi "k8s.io/client-go/1.5/tools/clientcmd/api"
)

type testLoader struct {
	ClientConfigLoader

	called bool
	config *clientcmdapi.Config
	err    error
}

func (l *testLoader) Load() (*clientcmdapi.Config, error) {
	l.called = true
	return l.config, l.err
}

type testClientConfig struct {
	config             *rest.Config
	namespace          string
	namespaceSpecified bool
	err                error
}

func (c *testClientConfig) RawConfig() (clientcmdapi.Config, error) {
	return clientcmdapi.Config{}, fmt.Errorf("unexpected call")
}
func (c *testClientConfig) ClientConfig() (*rest.Config, error) {
	return c.config, c.err
}
func (c *testClientConfig) Namespace() (string, bool, error) {
	return c.namespace, c.namespaceSpecified, c.err
}
func (c *testClientConfig) ConfigAccess() ConfigAccess {
	return nil
}

type testICC struct {
	testClientConfig

	possible bool
	called   bool
}

func (icc *testICC) Possible() bool {
	icc.called = true
	return icc.possible
}

func TestInClusterConfig(t *testing.T) {
	default1 := &DirectClientConfig{
		config:      *createValidTestConfig(),
		contextName: "clean",
		overrides:   &ConfigOverrides{},
	}
	invalidDefaultConfig := clientcmdapi.NewConfig()
	invalidDefaultConfig.Clusters["clean"] = &clientcmdapi.Cluster{
		Server: "http://localhost:8080",
	}
	invalidDefaultConfig.Contexts["other"] = &clientcmdapi.Context{
		Cluster: "clean",
	}
	invalidDefaultConfig.CurrentContext = "clean"

	defaultInvalid := &DirectClientConfig{
		config:    *invalidDefaultConfig,
		overrides: &ConfigOverrides{},
	}
	if _, err := defaultInvalid.ClientConfig(); err == nil || !IsConfigurationInvalid(err) {
		t.Fatal(err)
	}
	config1, err := default1.ClientConfig()
	if err != nil {
		t.Fatal(err)
	}
	config2 := &rest.Config{Host: "config2"}
	err1 := fmt.Errorf("unique error")

	testCases := map[string]struct {
		clientConfig  *testClientConfig
		icc           *testICC
		defaultConfig *DirectClientConfig

		checkedICC bool
		result     *rest.Config
		err        error
	}{
		"in-cluster checked on other error": {
			clientConfig: &testClientConfig{err: ErrEmptyConfig},
			icc:          &testICC{},

			checkedICC: true,
			result:     nil,
			err:        ErrEmptyConfig,
		},

		"in-cluster not checked on non-empty error": {
			clientConfig: &testClientConfig{err: ErrEmptyCluster},
			icc:          &testICC{},

			checkedICC: false,
			result:     nil,
			err:        ErrEmptyCluster,
		},

		"in-cluster checked when config is default": {
			defaultConfig: default1,
			clientConfig:  &testClientConfig{config: config1},
			icc:           &testICC{},

			checkedICC: true,
			result:     config1,
			err:        nil,
		},

		"in-cluster not checked when default config is invalid": {
			defaultConfig: defaultInvalid,
			clientConfig:  &testClientConfig{config: config1},
			icc:           &testICC{},

			checkedICC: false,
			result:     config1,
			err:        nil,
		},

		"in-cluster not checked when config is not equal to default": {
			defaultConfig: default1,
			clientConfig:  &testClientConfig{config: config2},
			icc:           &testICC{},

			checkedICC: false,
			result:     config2,
			err:        nil,
		},

		"in-cluster checked when config is not equal to default and error is empty": {
			clientConfig: &testClientConfig{config: config2, err: ErrEmptyConfig},
			icc:          &testICC{},

			checkedICC: true,
			result:     config2,
			err:        ErrEmptyConfig,
		},

		"in-cluster error returned when config is empty": {
			clientConfig: &testClientConfig{err: ErrEmptyConfig},
			icc: &testICC{
				possible: true,
				testClientConfig: testClientConfig{
					err: err1,
				},
			},

			checkedICC: true,
			result:     nil,
			err:        err1,
		},

		"in-cluster config returned when config is empty": {
			clientConfig: &testClientConfig{err: ErrEmptyConfig},
			icc: &testICC{
				possible: true,
				testClientConfig: testClientConfig{
					config: config2,
				},
			},

			checkedICC: true,
			result:     config2,
			err:        nil,
		},

		"in-cluster not checked when standard default is invalid": {
			defaultConfig: &DefaultClientConfig,
			clientConfig:  &testClientConfig{config: config2},
			icc:           &testICC{},

			checkedICC: false,
			result:     config2,
			err:        nil,
		},
	}

	for name, test := range testCases {
		c := &DeferredLoadingClientConfig{icc: test.icc}
		c.loader = &ClientConfigLoadingRules{DefaultClientConfig: test.defaultConfig}
		c.clientConfig = test.clientConfig

		cfg, err := c.ClientConfig()
		if test.icc.called != test.checkedICC {
			t.Errorf("%s: unexpected in-cluster-config call %t", name, test.icc.called)
		}
		if err != test.err || cfg != test.result {
			t.Errorf("%s: unexpected result: %v %#v", name, err, cfg)
		}
	}
}

func TestInClusterConfigNamespace(t *testing.T) {
	err1 := fmt.Errorf("unique error")

	testCases := map[string]struct {
		clientConfig *testClientConfig
		icc          *testICC

		checkedICC bool
		result     string
		ok         bool
		err        error
	}{
		"in-cluster checked on empty error": {
			clientConfig: &testClientConfig{err: ErrEmptyConfig},
			icc:          &testICC{},

			checkedICC: true,
			err:        ErrEmptyConfig,
		},

		"in-cluster not checked on non-empty error": {
			clientConfig: &testClientConfig{err: ErrEmptyCluster},
			icc:          &testICC{},

			err: ErrEmptyCluster,
		},

		"in-cluster checked when config is default": {
			clientConfig: &testClientConfig{},
			icc:          &testICC{},

			checkedICC: true,
		},

		"in-cluster not checked when config is not equal to default": {
			clientConfig: &testClientConfig{namespace: "test", namespaceSpecified: true},
			icc:          &testICC{},

			result: "test",
			ok:     true,
		},

		"in-cluster checked when namespace is not specified, but is defaulted": {
			clientConfig: &testClientConfig{namespace: "test", namespaceSpecified: false},
			icc:          &testICC{},

			checkedICC: true,
			result:     "test",
			ok:         false,
		},

		"in-cluster error returned when config is empty": {
			clientConfig: &testClientConfig{err: ErrEmptyConfig},
			icc: &testICC{
				possible: true,
				testClientConfig: testClientConfig{
					err: err1,
				},
			},

			checkedICC: true,
			err:        err1,
		},

		"in-cluster config returned when config is empty": {
			clientConfig: &testClientConfig{err: ErrEmptyConfig},
			icc: &testICC{
				possible: true,
				testClientConfig: testClientConfig{
					namespace:          "test",
					namespaceSpecified: true,
				},
			},

			checkedICC: true,
			result:     "test",
			ok:         true,
		},

		"in-cluster config returned when config is empty and namespace is defaulted but not explicitly set": {
			clientConfig: &testClientConfig{err: ErrEmptyConfig},
			icc: &testICC{
				possible: true,
				testClientConfig: testClientConfig{
					namespace:          "test",
					namespaceSpecified: false,
				},
			},

			checkedICC: true,
			result:     "test",
			ok:         false,
		},
	}

	for name, test := range testCases {
		c := &DeferredLoadingClientConfig{icc: test.icc}
		c.clientConfig = test.clientConfig

		ns, ok, err := c.Namespace()
		if test.icc.called != test.checkedICC {
			t.Errorf("%s: unexpected in-cluster-config call %t", name, test.icc.called)
		}
		if err != test.err || ns != test.result || ok != test.ok {
			t.Errorf("%s: unexpected result: %v %s %t", name, err, ns, ok)
		}
	}
}
