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

package framework

import (
	"path"

	"github.com/google/uuid"

	openapinamer "k8s.io/apiserver/pkg/endpoints/openapi"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/options"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	utilopenapi "k8s.io/apiserver/pkg/util/openapi"
	openapicommon "k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/spec3"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/generated/openapi"
)

const (
	UnprivilegedUserToken = "unprivileged-user"
)

// DefaultOpenAPIConfig returns an openapicommon.Config initialized to default values.
func DefaultOpenAPIConfig() *openapicommon.Config {
	openAPIConfig := genericapiserver.DefaultOpenAPIConfig(openapi.GetOpenAPIDefinitions, openapinamer.NewDefinitionNamer(legacyscheme.Scheme))
	openAPIConfig.Info = &spec.Info{
		InfoProps: spec.InfoProps{
			Title:   "Kubernetes",
			Version: "unversioned",
		},
	}
	openAPIConfig.DefaultResponse = &spec.Response{
		ResponseProps: spec.ResponseProps{
			Description: "Default Response.",
		},
	}
	openAPIConfig.GetDefinitions = utilopenapi.GetOpenAPIDefinitionsWithoutDisabledFeatures(openapi.GetOpenAPIDefinitions)

	return openAPIConfig
}

// DefaultOpenAPIV3Config returns an openapicommon.Config initialized to default values.
func DefaultOpenAPIV3Config() *openapicommon.OpenAPIV3Config {
	openAPIConfig := genericapiserver.DefaultOpenAPIV3Config(openapi.GetOpenAPIDefinitions, openapinamer.NewDefinitionNamer(legacyscheme.Scheme))
	openAPIConfig.Info = &spec.Info{
		InfoProps: spec.InfoProps{
			Title:   "Kubernetes",
			Version: "unversioned",
		},
	}
	openAPIConfig.DefaultResponse = &spec3.Response{
		ResponseProps: spec3.ResponseProps{
			Description: "Default Response.",
		},
	}
	openAPIConfig.GetDefinitions = utilopenapi.GetOpenAPIDefinitionsWithoutDisabledFeatures(openapi.GetOpenAPIDefinitions)

	return openAPIConfig
}

// DefaultEtcdOptions are the default EtcdOptions for use with integration tests.
func DefaultEtcdOptions() *options.EtcdOptions {
	// This causes the integration tests to exercise the etcd
	// prefix code, so please don't change without ensuring
	// sufficient coverage in other ways.
	etcdOptions := options.NewEtcdOptions(storagebackend.NewDefaultConfig(uuid.New().String(), nil))
	etcdOptions.StorageConfig.Transport.ServerList = []string{GetEtcdURL()}
	return etcdOptions
}

// SharedEtcd creates a storage config for a shared etcd instance, with a unique prefix.
func SharedEtcd() *storagebackend.Config {
	cfg := storagebackend.NewDefaultConfig(path.Join(uuid.New().String(), "registry"), nil)
	cfg.Transport.ServerList = []string{GetEtcdURL()}
	return cfg
}

// DefaultAPIServerFlags returns the default flags used to run kube-apiserver on tests
func DefaultTestServerFlags() []string {
	return []string{
		"--endpoint-reconciler-type=none",            // Disable Endpoints Reconciler so it does not keep failing trying to use 127.0.0.1 as a valid Endpoint.
		"--disable-admission-plugins=ServiceAccount", // Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	}
}
