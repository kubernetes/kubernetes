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

package discovery

import (
	"fmt"

	"k8s.io/client-go/pkg/api/unversioned"
	"k8s.io/client-go/pkg/util/sets"
	"k8s.io/client-go/pkg/version"
	"k8s.io/client-go/rest"
	// Import solely to initialize client auth plugins.
	_ "k8s.io/client-go/plugin/pkg/client/auth"
)

// MatchesServerVersion queries the server to compares the build version
// (git hash) of the client with the server's build version. It returns an error
// if it failed to contact the server or if the versions are not an exact match.
func MatchesServerVersion(client DiscoveryInterface, c *rest.Config) error {
	var err error
	if client == nil {
		client, err = NewDiscoveryClientForConfig(c)
		if err != nil {
			return err
		}
	}
	cVer := version.Get()
	sVer, err := client.ServerVersion()
	if err != nil {
		return fmt.Errorf("couldn't read version from server: %v\n", err)
	}
	// GitVersion includes GitCommit and GitTreeState, but best to be safe?
	if cVer.GitVersion != sVer.GitVersion || cVer.GitCommit != sVer.GitCommit || cVer.GitTreeState != sVer.GitTreeState {
		return fmt.Errorf("server version (%#v) differs from client version (%#v)!\n", sVer, cVer)
	}

	return nil
}

// NegotiateVersion queries the server's supported api versions to find
// a version that both client and server support.
// - If no version is provided, try registered client versions in order of
//   preference.
// - If version is provided, but not default config (explicitly requested via
//   commandline flag), and is unsupported by the server, print a warning to
//   stderr and try client's registered versions in order of preference.
// - If version is config default, and the server does not support it,
//   return an error.
func NegotiateVersion(client DiscoveryInterface, c *rest.Config, requestedGV *unversioned.GroupVersion, clientRegisteredGVs []unversioned.GroupVersion) (*unversioned.GroupVersion, error) {
	var err error
	if client == nil {
		client, err = NewDiscoveryClientForConfig(c)
		if err != nil {
			return nil, err
		}
	}
	clientVersions := sets.String{}
	for _, gv := range clientRegisteredGVs {
		clientVersions.Insert(gv.String())
	}
	groups, err := client.ServerGroups()
	if err != nil {
		// This is almost always a connection error, and higher level code should treat this as a generic error,
		// not a negotiation specific error.
		return nil, err
	}
	versions := unversioned.ExtractGroupVersions(groups)
	serverVersions := sets.String{}
	for _, v := range versions {
		serverVersions.Insert(v)
	}

	// If no version requested, use config version (may also be empty).
	// make a copy of the original so we don't risk mutating input here or in the returned value
	var preferredGV *unversioned.GroupVersion
	switch {
	case requestedGV != nil:
		t := *requestedGV
		preferredGV = &t
	case c.GroupVersion != nil:
		t := *c.GroupVersion
		preferredGV = &t
	}

	// If version explicitly requested verify that both client and server support it.
	// If server does not support warn, but try to negotiate a lower version.
	if preferredGV != nil {
		if !clientVersions.Has(preferredGV.String()) {
			return nil, fmt.Errorf("client does not support API version %q; client supported API versions: %v", preferredGV, clientVersions)

		}
		// If the server supports no versions, then we should just use the preferredGV
		// This can happen because discovery fails due to 403 Forbidden errors
		if len(serverVersions) == 0 {
			return preferredGV, nil
		}
		if serverVersions.Has(preferredGV.String()) {
			return preferredGV, nil
		}
		// If we are using an explicit config version the server does not support, fail.
		if (c.GroupVersion != nil) && (*preferredGV == *c.GroupVersion) {
			return nil, fmt.Errorf("server does not support API version %q", preferredGV)
		}
	}

	for _, clientGV := range clientRegisteredGVs {
		if serverVersions.Has(clientGV.String()) {
			// Version was not explicitly requested in command config (--api-version).
			// Ok to fall back to a supported version with a warning.
			// TODO: caesarxuchao: enable the warning message when we have
			// proper fix. Please refer to issue #14895.
			// if len(version) != 0 {
			// 	glog.Warningf("Server does not support API version '%s'. Falling back to '%s'.", version, clientVersion)
			// }
			t := clientGV
			return &t, nil
		}
	}
	return nil, fmt.Errorf("failed to negotiate an api version; server supports: %v, client supports: %v",
		serverVersions, clientVersions)
}
