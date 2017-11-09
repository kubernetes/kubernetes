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

package scheme

import (
	admission "k8s.io/api/admission/install"
	admissionregistration "k8s.io/api/admissionregistration/install"
	apps "k8s.io/api/apps/install"
	authentication "k8s.io/api/authentication/install"
	authorization "k8s.io/api/authorization/install"
	autoscaling "k8s.io/api/autoscaling/install"
	batch "k8s.io/api/batch/install"
	certificates "k8s.io/api/certificates/install"
	core "k8s.io/api/core/install"
	extensions "k8s.io/api/extensions/install"
	imagepolicy "k8s.io/api/imagepolicy/install"
	networking "k8s.io/api/networking/install"
	policy "k8s.io/api/policy/install"
	rbac "k8s.io/api/rbac/install"
	scheduling "k8s.io/api/scheduling/install"
	settings "k8s.io/api/settings/install"
	storage "k8s.io/api/storage/install"
)

func init() {
	admission.Install(GroupFactoryRegistry, Registry, Scheme)
	admissionregistration.Install(GroupFactoryRegistry, Registry, Scheme)
	core.Install(GroupFactoryRegistry, Registry, Scheme)
	apps.Install(GroupFactoryRegistry, Registry, Scheme)
	authentication.Install(GroupFactoryRegistry, Registry, Scheme)
	authorization.Install(GroupFactoryRegistry, Registry, Scheme)
	autoscaling.Install(GroupFactoryRegistry, Registry, Scheme)
	batch.Install(GroupFactoryRegistry, Registry, Scheme)
	certificates.Install(GroupFactoryRegistry, Registry, Scheme)
	extensions.Install(GroupFactoryRegistry, Registry, Scheme)
	imagepolicy.Install(GroupFactoryRegistry, Registry, Scheme)
	networking.Install(GroupFactoryRegistry, Registry, Scheme)
	policy.Install(GroupFactoryRegistry, Registry, Scheme)
	rbac.Install(GroupFactoryRegistry, Registry, Scheme)
	scheduling.Install(GroupFactoryRegistry, Registry, Scheme)
	settings.Install(GroupFactoryRegistry, Registry, Scheme)
	storage.Install(GroupFactoryRegistry, Registry, Scheme)
}
