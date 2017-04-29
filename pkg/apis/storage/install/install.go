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

// Package install installs the experimental API group, making it available as
// an option to all of the API encoding/decoding machinery.
package install

import (
	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/apis/storage/v1"
	"k8s.io/kubernetes/pkg/apis/storage/v1beta1"
)

func init() {
	Install(api.GroupFactoryRegistry, api.Registry, api.Scheme)
}

// Install registers the API group and adds types to a scheme
func Install(groupFactoryRegistry announced.APIGroupFactoryRegistry, registry *registered.APIRegistrationManager, scheme *runtime.Scheme) {
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName: storage.GroupName,
			/*
				DO NOT set the preferred storage version to v1 until Kubernetes 1.7. This will BREAK rolling
				cluster upgrades if you do:

				1. Start with 3 apiservers running 1.5
				2. Upgrade 1 apiserver to 1.6
				3. Someone sends a request to the 1.6 apiserver to create or update a storage class
				4. The 1.6 apiserver persists it as v1
				5. Anyone talking to the 1.5 apiservers that haven't been upgraded yet will break trying to
				   retrieve the storage class stored as v1.

				Once a cluster is 100% upgraded to 1.6, cluster administrators must run
				`cluster/update-storage-objects.sh` prior to upgrading to 1.7. This will update all
				storageclasses so they're stored in v1 format in etcd.
			*/
			VersionPreferenceOrder:     []string{v1beta1.SchemeGroupVersion.Version, v1.SchemeGroupVersion.Version},
			ImportPrefix:               "k8s.io/kubernetes/pkg/apis/storage",
			RootScopedKinds:            sets.NewString("StorageClass"),
			AddInternalObjectsToScheme: storage.AddToScheme,
		},
		announced.VersionToSchemeFunc{
			v1.SchemeGroupVersion.Version:      v1.AddToScheme,
			v1beta1.SchemeGroupVersion.Version: v1beta1.AddToScheme,
		},
	).Announce(groupFactoryRegistry).RegisterAndEnable(registry, scheme); err != nil {
		panic(err)
	}
}
