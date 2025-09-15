/*
Copyright 2015 The Kubernetes Authors.

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

package openshift

import (
	"testing"

	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/kubernetes/pkg/controlplane"
	controlplaneapiserver "k8s.io/kubernetes/pkg/controlplane/apiserver"
)

// This test references methods that OpenShift uses to customize the apiserver on startup, that
// are not referenced directly by an instance.
func TestApiserverExportsSymbols(t *testing.T) {
	_ = &controlplane.Config{
		ControlPlane: controlplaneapiserver.Config{
			Generic: &genericapiserver.Config{
				EnableMetrics: true,
			},
			Extra: controlplaneapiserver.Extra{
				EnableLogsSupport: false,
			},
		},
	}
	_ = &controlplane.Instance{
		ControlPlane: &controlplaneapiserver.Server{
			GenericAPIServer: &genericapiserver.GenericAPIServer{},
		},
	}
}
