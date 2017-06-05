/*
Copyright 2016 The Kubernetes Authors.

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

package apiserver

import (
	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/version"
	genericapiserver "k8s.io/apiserver/pkg/server"

	"k8s.io/apiserver-builder/pkg/builders"
	"k8s.io/client-go/pkg/api"
)

type Installer struct {
	GroupFactoryRegistry announced.APIGroupFactoryRegistry
	Registry             *registered.APIRegistrationManager
	Scheme               *runtime.Scheme
}

func (c *Config) Init() *Config {
	for _, builder := range builders.APIGroupBuilders {
		builder.Announce()
	}

	// we need to add the options to empty v1
	// TODO fix the server code to avoid this
	metav1.AddToGroupVersion(api.Scheme, schema.GroupVersion{Version: "v1"})

	// TODO: keep the generic ResourceDefinition server from wanting this
	unversioned := schema.GroupVersion{Group: "", Version: "v1"}
	api.Scheme.AddUnversionedTypes(unversioned,
		&metav1.Status{},
		&metav1.APIVersions{},
		&metav1.APIGroupList{},
		&metav1.APIGroup{},
		&metav1.APIResourceList{},
	)

	return c
}

type Config struct {
	GenericConfig *genericapiserver.Config
}

// Server contains state for a Kubernetes cluster master/api server.
type Server struct {
	GenericAPIServer *genericapiserver.GenericAPIServer
}

type completedConfig struct {
	*Config
}

// Complete fills in any fields not set that are required to have valid data. It's mutating the receiver.
func (c *Config) Complete() completedConfig {
	c.GenericConfig.Complete()

	c.GenericConfig.Version = &version.Info{
		Major: "1",
		Minor: "0",
	}

	return completedConfig{c}
}

func (c *Config) AddApi(builder *builders.APIGroupBuilder) *Config {
	builders.APIGroupBuilders = append(builders.APIGroupBuilders, builder)
	return c
}

// SkipComplete provides a way to construct a server instance without config completion.
func (c *Config) SkipComplete() completedConfig {
	return completedConfig{c}
}

// NewFunc returns a new instance of Server from the given config.
func (c completedConfig) New() (*Server, error) {
	genericServer, err := c.Config.GenericConfig.SkipComplete().New() // completion is done in Complete, no need for a second time
	if err != nil {
		return nil, err
	}

	s := &Server{
		GenericAPIServer: genericServer,
	}

	for _, builder := range builders.APIGroupBuilders {
		group := builder.Build(c.GenericConfig.RESTOptionsGetter)
		if err := s.GenericAPIServer.InstallAPIGroup(group); err != nil {
			return nil, err
		}
	}
	return s, nil
}
