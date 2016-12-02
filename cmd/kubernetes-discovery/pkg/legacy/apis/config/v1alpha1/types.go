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

package v1alpha1

// List of servers from which group versions should be summarized.
// This is used to represent the structure of the config file passed to discovery summarizer server.
type FederatedServerList struct {
	Servers []FederatedServer `json:"servers"`
}

// Information about each individual server, whose group versions needs to be summarized.
type FederatedServer struct {
	// The address that summarizer can reach to get discovery information from the server.
	// This can be hostname, hostname:port, IP or IP:port.
	ServerAddress string `json:"serverAddress"`
	// The list of paths where server exposes group version discovery information.
	// Summarizer will use these paths to figure out group versions supported by this server.
	GroupVersionDiscoveryPaths []GroupVersionDiscoveryPath `json:"groupVersionDiscoveryPaths"`
}

// Information about each group version discovery path that needs to be summarized.
type GroupVersionDiscoveryPath struct {
	// Path where the server exposes the discovery API to surface the group versions that it supports.
	Path string `json:"path"`

	// True if the path is for legacy group version.
	// (i.e the path returns unversioned.APIVersions instead of unversioned.APIGroupList)
	IsLegacy bool `json:"isLegacy"`
}
