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

package rest

import (
	"fmt"
	"sync"
	"time"

	"github.com/golang/glog"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/rbac"
	rbacapiv1alpha1 "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1"
	rbacclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/rbac/clusterrole"
	clusterrolepolicybased "k8s.io/kubernetes/pkg/registry/rbac/clusterrole/policybased"
	clusterrolestore "k8s.io/kubernetes/pkg/registry/rbac/clusterrole/storage"
	"k8s.io/kubernetes/pkg/registry/rbac/clusterrolebinding"
	clusterrolebindingpolicybased "k8s.io/kubernetes/pkg/registry/rbac/clusterrolebinding/policybased"
	clusterrolebindingstore "k8s.io/kubernetes/pkg/registry/rbac/clusterrolebinding/storage"
	"k8s.io/kubernetes/pkg/registry/rbac/role"
	rolepolicybased "k8s.io/kubernetes/pkg/registry/rbac/role/policybased"
	rolestore "k8s.io/kubernetes/pkg/registry/rbac/role/storage"
	"k8s.io/kubernetes/pkg/registry/rbac/rolebinding"
	rolebindingpolicybased "k8s.io/kubernetes/pkg/registry/rbac/rolebinding/policybased"
	rolebindingstore "k8s.io/kubernetes/pkg/registry/rbac/rolebinding/storage"
	rbacregistryvalidation "k8s.io/kubernetes/pkg/registry/rbac/validation"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac/bootstrappolicy"
)

type RESTStorageProvider struct {
	Authorizer authorizer.Authorizer
}

var _ genericapiserver.PostStartHookProvider = RESTStorageProvider{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(rbac.GroupName)

	if apiResourceConfigSource.AnyResourcesForVersionEnabled(rbacapiv1alpha1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[rbacapiv1alpha1.SchemeGroupVersion.Version] = p.v1alpha1Storage(apiResourceConfigSource, restOptionsGetter)
		apiGroupInfo.GroupMeta.GroupVersion = rbacapiv1alpha1.SchemeGroupVersion
	}

	return apiGroupInfo, true
}

func (p RESTStorageProvider) v1alpha1Storage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) map[string]rest.Storage {
	version := rbacapiv1alpha1.SchemeGroupVersion

	once := new(sync.Once)
	var (
		authorizationRuleResolver  rbacregistryvalidation.AuthorizationRuleResolver
		rolesStorage               rest.StandardStorage
		roleBindingsStorage        rest.StandardStorage
		clusterRolesStorage        rest.StandardStorage
		clusterRoleBindingsStorage rest.StandardStorage
	)

	initializeStorage := func() {
		once.Do(func() {
			rolesStorage = rolestore.NewREST(restOptionsGetter)
			roleBindingsStorage = rolebindingstore.NewREST(restOptionsGetter)
			clusterRolesStorage = clusterrolestore.NewREST(restOptionsGetter)
			clusterRoleBindingsStorage = clusterrolebindingstore.NewREST(restOptionsGetter)

			authorizationRuleResolver = rbacregistryvalidation.NewDefaultRuleResolver(
				role.AuthorizerAdapter{Registry: role.NewRegistry(rolesStorage)},
				rolebinding.AuthorizerAdapter{Registry: rolebinding.NewRegistry(roleBindingsStorage)},
				clusterrole.AuthorizerAdapter{Registry: clusterrole.NewRegistry(clusterRolesStorage)},
				clusterrolebinding.AuthorizerAdapter{Registry: clusterrolebinding.NewRegistry(clusterRoleBindingsStorage)},
			)
		})
	}

	storage := map[string]rest.Storage{}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("roles")) {
		initializeStorage()
		storage["roles"] = rolepolicybased.NewStorage(rolesStorage, authorizationRuleResolver)
	}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("rolebindings")) {
		initializeStorage()
		storage["rolebindings"] = rolebindingpolicybased.NewStorage(roleBindingsStorage, p.Authorizer, authorizationRuleResolver)
	}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("clusterroles")) {
		initializeStorage()
		storage["clusterroles"] = clusterrolepolicybased.NewStorage(clusterRolesStorage, authorizationRuleResolver)
	}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("clusterrolebindings")) {
		initializeStorage()
		storage["clusterrolebindings"] = clusterrolebindingpolicybased.NewStorage(clusterRoleBindingsStorage, p.Authorizer, authorizationRuleResolver)
	}
	return storage
}

func (p RESTStorageProvider) PostStartHook() (string, genericapiserver.PostStartHookFunc, error) {
	return "rbac/bootstrap-roles", PostStartHook, nil
}

func PostStartHook(hookContext genericapiserver.PostStartHookContext) error {
	// intializing roles is really important.  On some e2e runs, we've seen cases where etcd is down when the server
	// starts, the roles don't initialize, and nothing works.
	err := wait.Poll(1*time.Second, 30*time.Second, func() (done bool, err error) {
		clientset, err := rbacclient.NewForConfig(hookContext.LoopbackClientConfig)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("unable to initialize clusterroles: %v", err))
			return false, nil
		}

		existingClusterRoles, err := clientset.ClusterRoles().List(api.ListOptions{})
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("unable to initialize clusterroles: %v", err))
			return false, nil
		}
		// only initialized on empty etcd
		if len(existingClusterRoles.Items) == 0 {
			for _, clusterRole := range append(bootstrappolicy.ClusterRoles(), bootstrappolicy.ControllerRoles()...) {
				if _, err := clientset.ClusterRoles().Create(&clusterRole); err != nil {
					// don't fail on failures, try to create as many as you can
					utilruntime.HandleError(fmt.Errorf("unable to initialize clusterroles: %v", err))
					continue
				}
				glog.Infof("Created clusterrole.%s/%s", rbac.GroupName, clusterRole.Name)
			}
		}

		existingClusterRoleBindings, err := clientset.ClusterRoleBindings().List(api.ListOptions{})
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("unable to initialize clusterrolebindings: %v", err))
			return false, nil
		}
		// only initialized on empty etcd
		if len(existingClusterRoleBindings.Items) == 0 {
			for _, clusterRoleBinding := range append(bootstrappolicy.ClusterRoleBindings(), bootstrappolicy.ControllerRoleBindings()...) {
				if _, err := clientset.ClusterRoleBindings().Create(&clusterRoleBinding); err != nil {
					// don't fail on failures, try to create as many as you can
					utilruntime.HandleError(fmt.Errorf("unable to initialize clusterrolebindings: %v", err))
					continue
				}
				glog.Infof("Created clusterrolebinding.%s/%s", rbac.GroupName, clusterRoleBinding.Name)
			}
		}

		return true, nil
	})
	// if we're never able to make it through intialization, kill the API server
	if err != nil {
		return fmt.Errorf("unable to initialize roles: %v", err)
	}

	return nil
}

func (p RESTStorageProvider) GroupName() string {
	return rbac.GroupName
}
