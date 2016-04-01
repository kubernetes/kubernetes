/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package config

import "k8s.io/kubernetes/pkg/api"

// APIServerConfig is a structure used to configure a GenericAPIServer.
type APIServerConfig struct {
	// StorageVersions is a map between groups and their storage versions
	StorageVersions map[string]string
	// allow downstream consumers to disable the core controller loops
	EnableLogsSupport bool
	EnableUISupport   bool
	// Allow downstream consumers to disable swagger.
	// This includes returning the generated swagger spec at /swaggerapi and swagger ui at /swagger-ui.
	EnableSwaggerSupport bool
	// Allow downstream consumers to disable swagger ui.
	// Note that this is ignored if either EnableSwaggerSupport or EnableUISupport is false.
	EnableSwaggerUI bool
	// allow downstream consumers to disable the index route
	EnableIndex           bool
	EnableProfiling       bool
	EnableWatchCache      bool
	APIPrefix             string
	APIGroupPrefix        string
	CorsAllowedOriginList []string
	// TODO(roberthbailey): Remove once the server no longer supports http basic auth.
	SupportsBasicAuth      bool
	MasterServiceNamespace string
	// If specified, requests will be allocated a random timeout between this value, and twice this value.
	// Note that it is up to the request handlers to ignore or honor this timeout. In seconds.
	MinRequestTimeout int
	// Number of masters running; all masters must be started with the
	// same value for this field. (Numbers > 1 currently untested.)
	MasterCount int
	// The port on PublicAddress where a read-write server will be installed.
	// Defaults to 6443 if not set.
	ReadWritePort int
	// ExternalHost is the host name to use for external (public internet) facing URLs (e.g. Swagger)
	ExternalHost string
	// Port for the apiserver service.
	ServiceReadWritePort int
	// Additional ports to be exposed on the GenericAPIServer service
	// extraServicePorts is injectable in the event that more ports
	// (other than the default 443/tcp) are exposed on the GenericAPIServer
	// and those ports need to be load balanced by the GenericAPIServer
	// service because this pkg is linked by out-of-tree projects
	// like openshift which want to use the GenericAPIServer but also do
	// more stuff.
	ExtraServicePorts []api.ServicePort
	// Additional ports to be exposed on the GenericAPIServer endpoints
	// Port names should align with ports defined in ExtraServicePorts
	ExtraEndpointPorts        []api.EndpointPort
	KubernetesServiceNodePort int
}
