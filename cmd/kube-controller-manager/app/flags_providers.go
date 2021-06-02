// +build !providerless

/*
Copyright 2019 The Kubernetes Authors.

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

package app

import (
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/cli/globalflag"
)

func registerLegacyGlobalFlags(namedFlagSets cliflag.NamedFlagSets) {
	// hoist this flag from the global flagset to preserve the commandline until
	// the gce cloudprovider is removed.
	globalflag.Register(namedFlagSets.FlagSet("generic"), "cloud-provider-gce-lb-src-cidrs")
	namedFlagSets.FlagSet("generic").MarkDeprecated("cloud-provider-gce-lb-src-cidrs", "This flag will be removed once the GCE Cloud Provider is removed from kube-controller-manager")
}
