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

package clientset

// These imports are the API groups the client will support.
import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	core "k8s.io/kubernetes/pkg/api/install"
	apps "k8s.io/kubernetes/pkg/apis/apps/install"
	authentication "k8s.io/kubernetes/pkg/apis/authentication/install"
	authorization "k8s.io/kubernetes/pkg/apis/authorization/install"
	autoscaling "k8s.io/kubernetes/pkg/apis/autoscaling/install"
	batch "k8s.io/kubernetes/pkg/apis/batch/install"
	certificates "k8s.io/kubernetes/pkg/apis/certificates/install"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/install"
	policy "k8s.io/kubernetes/pkg/apis/policy/install"
	rbac "k8s.io/kubernetes/pkg/apis/rbac/install"
	settings "k8s.io/kubernetes/pkg/apis/settings/install"
	storage "k8s.io/kubernetes/pkg/apis/storage/install"
)

func init() {
	core.Install(api.GroupFactoryRegistry, api.Registry, api.Scheme)
	apps.Install(api.GroupFactoryRegistry, api.Registry, api.Scheme)
	authentication.Install(api.GroupFactoryRegistry, api.Registry, api.Scheme)
	authorization.Install(api.GroupFactoryRegistry, api.Registry, api.Scheme)
	autoscaling.Install(api.GroupFactoryRegistry, api.Registry, api.Scheme)
	batch.Install(api.GroupFactoryRegistry, api.Registry, api.Scheme)
	certificates.Install(api.GroupFactoryRegistry, api.Registry, api.Scheme)
	extensions.Install(api.GroupFactoryRegistry, api.Registry, api.Scheme)
	policy.Install(api.GroupFactoryRegistry, api.Registry, api.Scheme)
	rbac.Install(api.GroupFactoryRegistry, api.Registry, api.Scheme)
	settings.Install(api.GroupFactoryRegistry, api.Registry, api.Scheme)
	storage.Install(api.GroupFactoryRegistry, api.Registry, api.Scheme)

	if missingVersions := api.Registry.ValidateEnvRequestedVersions(); len(missingVersions) != 0 {
		panic(fmt.Sprintf("KUBE_API_VERSIONS contains versions that are not installed: %q.", missingVersions))
	}
}
