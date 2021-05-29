/*
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package meta

import (
	"reflect"

	alpha "google.golang.org/api/compute/v0.alpha"
	beta "google.golang.org/api/compute/v0.beta"
	ga "google.golang.org/api/compute/v1"
)

// Version of the API (ga, alpha, beta).
type Version string

const (
	// NoGet prevents the Get() method from being generated.
	NoGet = 1 << iota
	// NoList prevents the List() method from being generated.
	NoList = 1 << iota
	// NoDelete prevents the Delete() method from being generated.
	NoDelete = 1 << iota
	// NoInsert prevents the Insert() method from being generated.
	NoInsert = 1 << iota
	// CustomOps specifies that an empty interface xxxOps will be generated to
	// enable custom method calls to be attached to the generated service
	// interface.
	CustomOps = 1 << iota
	// AggregatedList will generated a method for AggregatedList().
	AggregatedList = 1 << iota

	// ReadOnly specifies that the given resource is read-only and should not
	// have insert() or delete() methods generated for the wrapper.
	ReadOnly = NoDelete | NoInsert

	// VersionGA is the API version in compute.v1.
	VersionGA Version = "ga"
	// VersionAlpha is the API version in computer.v0.alpha.
	VersionAlpha Version = "alpha"
	// VersionBeta is the API version in computer.v0.beta.
	VersionBeta Version = "beta"
)

// AllVersions is a list of all versions of the GCE API.
var AllVersions = []Version{
	VersionGA,
	VersionAlpha,
	VersionBeta,
}

// AllServices are a list of all the services to generate code for. Keep
// this list in lexiographical order by object type.
var AllServices = []*ServiceInfo{
	{
		Object:      "Address",
		Service:     "Addresses",
		Resource:    "addresses",
		keyType:     Regional,
		serviceType: reflect.TypeOf(&ga.AddressesService{}),
		options:     AggregatedList,
	},
	{
		Object:      "Address",
		Service:     "Addresses",
		Resource:    "addresses",
		version:     VersionAlpha,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&alpha.AddressesService{}),
		options:     AggregatedList,
	},
	{
		Object:      "Address",
		Service:     "Addresses",
		Resource:    "addresses",
		version:     VersionBeta,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&beta.AddressesService{}),
		options:     AggregatedList,
	},
	{
		Object:      "Address",
		Service:     "GlobalAddresses",
		Resource:    "addresses",
		version:     VersionAlpha,
		keyType:     Global,
		serviceType: reflect.TypeOf(&alpha.GlobalAddressesService{}),
	},
	{
		Object:      "Address",
		Service:     "GlobalAddresses",
		Resource:    "addresses",
		keyType:     Global,
		serviceType: reflect.TypeOf(&ga.GlobalAddressesService{}),
	},
	{
		Object:      "BackendService",
		Service:     "BackendServices",
		Resource:    "backendServices",
		keyType:     Global,
		serviceType: reflect.TypeOf(&ga.BackendServicesService{}),
		additionalMethods: []string{
			"GetHealth",
			"Patch",
			"Update",
		},
	},
	{
		Object:      "BackendService",
		Service:     "BackendServices",
		Resource:    "backendServices",
		version:     VersionBeta,
		keyType:     Global,
		serviceType: reflect.TypeOf(&beta.BackendServicesService{}),
		additionalMethods: []string{
			"Update",
			"SetSecurityPolicy",
		},
	},
	{
		Object:      "BackendService",
		Service:     "BackendServices",
		Resource:    "backendServices",
		version:     VersionAlpha,
		keyType:     Global,
		serviceType: reflect.TypeOf(&alpha.BackendServicesService{}),
		additionalMethods: []string{
			"Update",
			"SetSecurityPolicy",
		},
	},
	{
		Object:      "BackendService",
		Service:     "RegionBackendServices",
		Resource:    "backendServices",
		version:     VersionGA,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&ga.RegionBackendServicesService{}),
		additionalMethods: []string{
			"GetHealth",
			"Update",
		},
	},
	{
		Object:      "BackendService",
		Service:     "RegionBackendServices",
		Resource:    "backendServices",
		version:     VersionAlpha,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&alpha.RegionBackendServicesService{}),
		additionalMethods: []string{
			"GetHealth",
			"Update",
		},
	},
	{
		Object:      "BackendService",
		Service:     "RegionBackendServices",
		Resource:    "backendServices",
		version:     VersionBeta,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&beta.RegionBackendServicesService{}),
		additionalMethods: []string{
			"GetHealth",
			"Update",
		},
	},
	{
		Object:      "Disk",
		Service:     "Disks",
		Resource:    "disks",
		keyType:     Zonal,
		serviceType: reflect.TypeOf(&ga.DisksService{}),
		additionalMethods: []string{
			"Resize",
		},
	},
	{
		Object:      "Disk",
		Service:     "RegionDisks",
		Resource:    "disks",
		version:     VersionGA,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&ga.RegionDisksService{}),
		additionalMethods: []string{
			"Resize",
		},
	},
	{
		Object:      "Firewall",
		Service:     "Firewalls",
		Resource:    "firewalls",
		version:     VersionAlpha,
		keyType:     Global,
		serviceType: reflect.TypeOf(&alpha.FirewallsService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "Firewall",
		Service:     "Firewalls",
		Resource:    "firewalls",
		version:     VersionBeta,
		keyType:     Global,
		serviceType: reflect.TypeOf(&beta.FirewallsService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "Firewall",
		Service:     "Firewalls",
		Resource:    "firewalls",
		keyType:     Global,
		serviceType: reflect.TypeOf(&ga.FirewallsService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "ForwardingRule",
		Service:     "ForwardingRules",
		Resource:    "forwardingRules",
		keyType:     Regional,
		serviceType: reflect.TypeOf(&ga.ForwardingRulesService{}),
		additionalMethods: []string{
			"SetTarget",
		},
	},
	{
		Object:      "ForwardingRule",
		Service:     "ForwardingRules",
		Resource:    "forwardingRules",
		version:     VersionAlpha,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&alpha.ForwardingRulesService{}),
		additionalMethods: []string{
			"SetTarget",
		},
	},
	{
		Object:      "ForwardingRule",
		Service:     "ForwardingRules",
		Resource:    "forwardingRules",
		version:     VersionBeta,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&beta.ForwardingRulesService{}),
		additionalMethods: []string{
			"SetTarget",
		},
	},
	{
		Object:      "ForwardingRule",
		Service:     "GlobalForwardingRules",
		Resource:    "forwardingRules",
		version:     VersionAlpha,
		keyType:     Global,
		serviceType: reflect.TypeOf(&alpha.GlobalForwardingRulesService{}),
		additionalMethods: []string{
			"SetTarget",
		},
	},
	{
		Object:      "ForwardingRule",
		Service:     "GlobalForwardingRules",
		Resource:    "forwardingRules",
		version:     VersionBeta,
		keyType:     Global,
		serviceType: reflect.TypeOf(&beta.GlobalForwardingRulesService{}),
		additionalMethods: []string{
			"SetTarget",
		},
	},
	{
		Object:      "ForwardingRule",
		Service:     "GlobalForwardingRules",
		Resource:    "forwardingRules",
		keyType:     Global,
		serviceType: reflect.TypeOf(&ga.GlobalForwardingRulesService{}),
		additionalMethods: []string{
			"SetTarget",
		},
	},
	{
		Object:      "HealthCheck",
		Service:     "HealthChecks",
		Resource:    "healthChecks",
		keyType:     Global,
		serviceType: reflect.TypeOf(&ga.HealthChecksService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "HealthCheck",
		Service:     "HealthChecks",
		Resource:    "healthChecks",
		version:     VersionAlpha,
		keyType:     Global,
		serviceType: reflect.TypeOf(&alpha.HealthChecksService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "HealthCheck",
		Service:     "HealthChecks",
		Resource:    "healthChecks",
		version:     VersionBeta,
		keyType:     Global,
		serviceType: reflect.TypeOf(&beta.HealthChecksService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "HealthCheck",
		Service:     "RegionHealthChecks",
		Resource:    "healthChecks",
		version:     VersionAlpha,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&alpha.RegionHealthChecksService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "HealthCheck",
		Service:     "RegionHealthChecks",
		Resource:    "healthChecks",
		version:     VersionBeta,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&beta.RegionHealthChecksService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "HealthCheck",
		Service:     "RegionHealthChecks",
		Resource:    "healthChecks",
		version:     VersionGA,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&ga.RegionHealthChecksService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "HttpHealthCheck",
		Service:     "HttpHealthChecks",
		Resource:    "httpHealthChecks",
		keyType:     Global,
		serviceType: reflect.TypeOf(&ga.HttpHealthChecksService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "HttpsHealthCheck",
		Service:     "HttpsHealthChecks",
		Resource:    "httpsHealthChecks",
		keyType:     Global,
		serviceType: reflect.TypeOf(&ga.HttpsHealthChecksService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "InstanceGroup",
		Service:     "InstanceGroups",
		Resource:    "instanceGroups",
		keyType:     Zonal,
		serviceType: reflect.TypeOf(&ga.InstanceGroupsService{}),
		additionalMethods: []string{
			"AddInstances",
			"ListInstances",
			"RemoveInstances",
			"SetNamedPorts",
		},
	},
	{
		Object:      "Instance",
		Service:     "Instances",
		Resource:    "instances",
		keyType:     Zonal,
		serviceType: reflect.TypeOf(&ga.InstancesService{}),
		additionalMethods: []string{
			"AttachDisk",
			"DetachDisk",
		},
	},
	{
		Object:      "Instance",
		Service:     "Instances",
		Resource:    "instances",
		version:     VersionBeta,
		keyType:     Zonal,
		serviceType: reflect.TypeOf(&beta.InstancesService{}),
		additionalMethods: []string{
			"AttachDisk",
			"DetachDisk",
			"UpdateNetworkInterface",
		},
	},
	{
		Object:      "Instance",
		Service:     "Instances",
		Resource:    "instances",
		version:     VersionAlpha,
		keyType:     Zonal,
		serviceType: reflect.TypeOf(&alpha.InstancesService{}),
		additionalMethods: []string{
			"AttachDisk",
			"DetachDisk",
			"UpdateNetworkInterface",
		},
	},
	{
		Object:      "Network",
		Service:     "Networks",
		Resource:    "networks",
		version:     VersionAlpha,
		keyType:     Global,
		serviceType: reflect.TypeOf(&alpha.NetworksService{}),
	},
	{
		Object:      "Network",
		Service:     "Networks",
		Resource:    "networks",
		version:     VersionBeta,
		keyType:     Global,
		serviceType: reflect.TypeOf(&alpha.NetworksService{}),
	},
	{
		Object:      "Network",
		Service:     "Networks",
		Resource:    "networks",
		version:     VersionGA,
		keyType:     Global,
		serviceType: reflect.TypeOf(&alpha.NetworksService{}),
	},
	{
		Object:      "NetworkEndpointGroup",
		Service:     "NetworkEndpointGroups",
		Resource:    "networkEndpointGroups",
		version:     VersionAlpha,
		keyType:     Zonal,
		serviceType: reflect.TypeOf(&alpha.NetworkEndpointGroupsService{}),
		additionalMethods: []string{
			"AttachNetworkEndpoints",
			"DetachNetworkEndpoints",
			"ListNetworkEndpoints",
		},
		options: AggregatedList,
	},
	{
		Object:      "NetworkEndpointGroup",
		Service:     "NetworkEndpointGroups",
		Resource:    "networkEndpointGroups",
		version:     VersionBeta,
		keyType:     Zonal,
		serviceType: reflect.TypeOf(&beta.NetworkEndpointGroupsService{}),
		additionalMethods: []string{
			"AttachNetworkEndpoints",
			"DetachNetworkEndpoints",
			"ListNetworkEndpoints",
		},
		options: AggregatedList,
	},
	{
		Object:      "NetworkEndpointGroup",
		Service:     "NetworkEndpointGroups",
		Resource:    "networkEndpointGroups",
		version:     VersionGA,
		keyType:     Zonal,
		serviceType: reflect.TypeOf(&ga.NetworkEndpointGroupsService{}),
		additionalMethods: []string{
			"AttachNetworkEndpoints",
			"DetachNetworkEndpoints",
			"ListNetworkEndpoints",
		},
		options: AggregatedList,
	},
	{
		Object:   "Project",
		Service:  "Projects",
		Resource: "projects",
		keyType:  Global,
		// Generate only the stub with no methods.
		options:     NoGet | NoList | NoInsert | NoDelete | CustomOps,
		serviceType: reflect.TypeOf(&ga.ProjectsService{}),
	},
	{
		Object:      "Region",
		Service:     "Regions",
		Resource:    "regions",
		keyType:     Global,
		options:     ReadOnly,
		serviceType: reflect.TypeOf(&ga.RegionsService{}),
	},
	{
		Object:      "Route",
		Service:     "Routes",
		Resource:    "routes",
		keyType:     Global,
		serviceType: reflect.TypeOf(&ga.RoutesService{}),
	},
	{
		Object:      "SecurityPolicy",
		Service:     "SecurityPolicies",
		Resource:    "securityPolicies",
		version:     VersionBeta,
		keyType:     Global,
		serviceType: reflect.TypeOf(&beta.SecurityPoliciesService{}),
		additionalMethods: []string{
			"AddRule",
			"GetRule",
			"Patch",
			"PatchRule",
			"RemoveRule",
		},
	},
	{
		Object:      "SslCertificate",
		Service:     "SslCertificates",
		Resource:    "sslCertificates",
		keyType:     Global,
		serviceType: reflect.TypeOf(&ga.SslCertificatesService{}),
	},
	{
		Object:      "SslCertificate",
		Service:     "SslCertificates",
		Resource:    "sslCertificates",
		version:     VersionBeta,
		keyType:     Global,
		serviceType: reflect.TypeOf(&beta.SslCertificatesService{}),
	},
	{
		Object:      "SslCertificate",
		Service:     "SslCertificates",
		Resource:    "sslCertificates",
		version:     VersionAlpha,
		keyType:     Global,
		serviceType: reflect.TypeOf(&alpha.SslCertificatesService{}),
	},
	{
		Object:      "SslCertificate",
		Service:     "RegionSslCertificates",
		Resource:    "sslCertificates",
		version:     VersionAlpha,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&alpha.RegionSslCertificatesService{}),
	},
	{
		Object:      "SslCertificate",
		Service:     "RegionSslCertificates",
		Resource:    "sslCertificates",
		version:     VersionBeta,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&beta.RegionSslCertificatesService{}),
	},
	{
		Object:      "SslCertificate",
		Service:     "RegionSslCertificates",
		Resource:    "sslCertificates",
		version:     VersionGA,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&ga.RegionSslCertificatesService{}),
	},
	{
		Object:      "SslPolicy",
		Service:     "SslPolicies",
		Resource:    "sslPolicies",
		keyType:     Global,
		serviceType: reflect.TypeOf(&ga.SslPoliciesService{}),
		options:     NoList, // List() naming convention is different in GCE API for this resource
	},
	{
		Object:      "Subnetwork",
		Service:     "Subnetworks",
		Resource:    "subnetworks",
		version:     VersionAlpha,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&alpha.SubnetworksService{}),
	},
	{
		Object:      "Subnetwork",
		Service:     "Subnetworks",
		Resource:    "subnetworks",
		version:     VersionBeta,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&alpha.SubnetworksService{}),
	},
	{
		Object:      "Subnetwork",
		Service:     "Subnetworks",
		Resource:    "subnetworks",
		version:     VersionGA,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&alpha.SubnetworksService{}),
	},
	{
		Object:      "TargetHttpProxy",
		Service:     "TargetHttpProxies",
		Resource:    "targetHttpProxies",
		version:     VersionAlpha,
		keyType:     Global,
		serviceType: reflect.TypeOf(&alpha.TargetHttpProxiesService{}),
		additionalMethods: []string{
			"SetUrlMap",
		},
	},
	{
		Object:      "TargetHttpProxy",
		Service:     "TargetHttpProxies",
		Resource:    "targetHttpProxies",
		version:     VersionBeta,
		keyType:     Global,
		serviceType: reflect.TypeOf(&beta.TargetHttpProxiesService{}),
		additionalMethods: []string{
			"SetUrlMap",
		},
	},
	{
		Object:      "TargetHttpProxy",
		Service:     "TargetHttpProxies",
		Resource:    "targetHttpProxies",
		keyType:     Global,
		serviceType: reflect.TypeOf(&ga.TargetHttpProxiesService{}),
		additionalMethods: []string{
			"SetUrlMap",
		},
	},
	{
		Object:      "TargetHttpProxy",
		Service:     "RegionTargetHttpProxies",
		Resource:    "targetHttpProxies",
		version:     VersionAlpha,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&alpha.RegionTargetHttpProxiesService{}),
		additionalMethods: []string{
			"SetUrlMap",
		},
	},
	{
		Object:      "TargetHttpProxy",
		Service:     "RegionTargetHttpProxies",
		Resource:    "targetHttpProxies",
		version:     VersionBeta,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&beta.RegionTargetHttpProxiesService{}),
		additionalMethods: []string{
			"SetUrlMap",
		},
	},
	{
		Object:      "TargetHttpProxy",
		Service:     "RegionTargetHttpProxies",
		Resource:    "targetHttpProxies",
		version:     VersionGA,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&ga.RegionTargetHttpProxiesService{}),
		additionalMethods: []string{
			"SetUrlMap",
		},
	},
	{
		Object:      "TargetHttpsProxy",
		Service:     "TargetHttpsProxies",
		Resource:    "targetHttpsProxies",
		keyType:     Global,
		serviceType: reflect.TypeOf(&ga.TargetHttpsProxiesService{}),
		additionalMethods: []string{
			"SetSslCertificates",
			"SetSslPolicy",
			"SetUrlMap",
		},
	},
	{
		Object:      "TargetHttpsProxy",
		Service:     "TargetHttpsProxies",
		Resource:    "targetHttpsProxies",
		version:     VersionAlpha,
		keyType:     Global,
		serviceType: reflect.TypeOf(&alpha.TargetHttpsProxiesService{}),
		additionalMethods: []string{
			"SetSslCertificates",
			"SetSslPolicy",
			"SetUrlMap",
		},
	},
	{
		Object:      "TargetHttpsProxy",
		Service:     "TargetHttpsProxies",
		Resource:    "targetHttpsProxies",
		version:     VersionBeta,
		keyType:     Global,
		serviceType: reflect.TypeOf(&beta.TargetHttpsProxiesService{}),
		additionalMethods: []string{
			"SetSslCertificates",
			"SetSslPolicy",
			"SetUrlMap",
		},
	},
	{
		Object:      "TargetHttpsProxy",
		Service:     "RegionTargetHttpsProxies",
		Resource:    "targetHttpsProxies",
		version:     VersionAlpha,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&alpha.RegionTargetHttpsProxiesService{}),
		additionalMethods: []string{
			"SetSslCertificates",
			"SetUrlMap",
		},
	},
	{
		Object:      "TargetHttpsProxy",
		Service:     "RegionTargetHttpsProxies",
		Resource:    "targetHttpsProxies",
		version:     VersionBeta,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&beta.RegionTargetHttpsProxiesService{}),
		additionalMethods: []string{
			"SetSslCertificates",
			"SetUrlMap",
		},
	},
	{
		Object:      "TargetHttpsProxy",
		Service:     "RegionTargetHttpsProxies",
		Resource:    "targetHttpsProxies",
		version:     VersionGA,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&ga.RegionTargetHttpsProxiesService{}),
		additionalMethods: []string{
			"SetSslCertificates",
			"SetUrlMap",
		},
	},
	{
		Object:      "TargetPool",
		Service:     "TargetPools",
		Resource:    "targetPools",
		keyType:     Regional,
		serviceType: reflect.TypeOf(&ga.TargetPoolsService{}),
		additionalMethods: []string{
			"AddInstance",
			"RemoveInstance",
		},
	},
	{
		Object:      "UrlMap",
		Service:     "UrlMaps",
		Resource:    "urlMaps",
		version:     VersionAlpha,
		keyType:     Global,
		serviceType: reflect.TypeOf(&alpha.UrlMapsService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "UrlMap",
		Service:     "UrlMaps",
		Resource:    "urlMaps",
		version:     VersionBeta,
		keyType:     Global,
		serviceType: reflect.TypeOf(&beta.UrlMapsService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "UrlMap",
		Service:     "UrlMaps",
		Resource:    "urlMaps",
		keyType:     Global,
		serviceType: reflect.TypeOf(&ga.UrlMapsService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "UrlMap",
		Service:     "RegionUrlMaps",
		Resource:    "urlMaps",
		version:     VersionAlpha,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&alpha.RegionUrlMapsService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "UrlMap",
		Service:     "RegionUrlMaps",
		Resource:    "urlMaps",
		version:     VersionBeta,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&beta.RegionUrlMapsService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "UrlMap",
		Service:     "RegionUrlMaps",
		Resource:    "urlMaps",
		version:     VersionGA,
		keyType:     Regional,
		serviceType: reflect.TypeOf(&ga.RegionUrlMapsService{}),
		additionalMethods: []string{
			"Update",
		},
	},
	{
		Object:      "Zone",
		Service:     "Zones",
		Resource:    "zones",
		keyType:     Global,
		options:     ReadOnly,
		serviceType: reflect.TypeOf(&ga.ZonesService{}),
	},
}
