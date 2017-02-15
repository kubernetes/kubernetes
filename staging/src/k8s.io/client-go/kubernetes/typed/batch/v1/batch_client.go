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

package v1

import (
	fmt "fmt"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	rest "k8s.io/client-go/rest"
	"k8s.io/apimachinery/pkg/runtime"
)

type BatchV1Interface interface {
	RESTClient() rest.Interface
	JobsGetter
}

// BatchV1Client is used to interact with features provided by the batch group.
type BatchV1Client struct {
	restClient rest.Interface
	parameterCodec runtime.ParameterCodec
}

func (c *BatchV1Client) Jobs(namespace string) JobInterface {
	return newJobs(c, namespace, c.parameterCodec)
}

// NewForConfig creates a new BatchV1Client for the given config.
func NewForConfig(c *rest.Config) (*BatchV1Client, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	client, err := rest.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &BatchV1Client{client, c.ParameterCodec}, nil
}

// NewForConfigOrDie creates a new BatchV1Client for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *rest.Config) *BatchV1Client {
	client, err := NewForConfig(c)
	if err != nil {
		panic(err)
	}
	return client
}

// New creates a new BatchV1Client for the given RESTClient. Only used and meant for testing.
func New(c rest.Interface, parameterCodec runtime.ParameterCodec) *BatchV1Client {
	return &BatchV1Client{c, parameterCodec}
}

func setConfigDefaults(config *rest.Config) error {
	gv, err := schema.ParseGroupVersion("batch/v1")
	if err != nil {
		return err
	}

	if config.NegotiatedSerializer == nil {
		return fmt.Errorf("expected non-nil NegotiatedSerializer for %v client", gv)
	}
	if config.ParameterCodec == nil {
		return fmt.Errorf("expected non-nil ParameterCodec for %v client", gv)
	}

	config.APIPath = "/apis"
	if config.UserAgent == "" {
		config.UserAgent = rest.DefaultKubernetesUserAgent()
	}
	copyGroupVersion := gv
	config.GroupVersion = &copyGroupVersion

	return nil
}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *BatchV1Client) RESTClient() rest.Interface {
	if c == nil {
		return nil
	}
	return c.restClient
}
