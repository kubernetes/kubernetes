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
	announced "k8s.io/apimachinery/pkg/apimachinery/announced"
	registered "k8s.io/apimachinery/pkg/apimachinery/registered"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	serializer "k8s.io/apimachinery/pkg/runtime/serializer"
	core "k8s.io/kubernetes/pkg/api/install"
	admissionregistration "k8s.io/kubernetes/pkg/apis/admissionregistration/install"
	apps "k8s.io/kubernetes/pkg/apis/apps/install"
	authentication "k8s.io/kubernetes/pkg/apis/authentication/install"
	authorization "k8s.io/kubernetes/pkg/apis/authorization/install"
	autoscaling "k8s.io/kubernetes/pkg/apis/autoscaling/install"
	batch "k8s.io/kubernetes/pkg/apis/batch/install"
	certificates "k8s.io/kubernetes/pkg/apis/certificates/install"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/install"
	networking "k8s.io/kubernetes/pkg/apis/networking/install"
	policy "k8s.io/kubernetes/pkg/apis/policy/install"
	rbac "k8s.io/kubernetes/pkg/apis/rbac/install"
	scheduling "k8s.io/kubernetes/pkg/apis/scheduling/install"
	settings "k8s.io/kubernetes/pkg/apis/settings/install"
	storage "k8s.io/kubernetes/pkg/apis/storage/install"
	os "os"
)

var Scheme = runtime.NewScheme()
var Codecs = serializer.NewCodecFactory(Scheme)
var ParameterCodec = runtime.NewParameterCodec(Scheme)

var Registry = registered.NewOrDie(os.Getenv("KUBE_API_VERSIONS"))
var GroupFactoryRegistry = make(announced.APIGroupFactoryRegistry)

func init() {
	v1.AddToGroupVersion(Scheme, schema.GroupVersion{Version: "v1"})
	Install(GroupFactoryRegistry, Registry, Scheme)
}

// Install registers the API group and adds types to a scheme
func Install(groupFactoryRegistry announced.APIGroupFactoryRegistry, registry *registered.APIRegistrationManager, scheme *runtime.Scheme) {
	admissionregistration.Install(groupFactoryRegistry, registry, scheme)
	core.Install(groupFactoryRegistry, registry, scheme)
	apps.Install(groupFactoryRegistry, registry, scheme)
	authentication.Install(groupFactoryRegistry, registry, scheme)
	authorization.Install(groupFactoryRegistry, registry, scheme)
	autoscaling.Install(groupFactoryRegistry, registry, scheme)
	batch.Install(groupFactoryRegistry, registry, scheme)
	certificates.Install(groupFactoryRegistry, registry, scheme)
	extensions.Install(groupFactoryRegistry, registry, scheme)
	networking.Install(groupFactoryRegistry, registry, scheme)
	policy.Install(groupFactoryRegistry, registry, scheme)
	rbac.Install(groupFactoryRegistry, registry, scheme)
	scheduling.Install(groupFactoryRegistry, registry, scheme)
	settings.Install(groupFactoryRegistry, registry, scheme)
	storage.Install(groupFactoryRegistry, registry, scheme)

	ExtraInstall(groupFactoryRegistry, registry, scheme)
}
