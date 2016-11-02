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

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/rbac"
	rbacapiv1alpha1 "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1"
	rbacvalidation "k8s.io/kubernetes/pkg/apis/rbac/validation"
	rbacclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/registry/rbac/clusterrole"
	clusterroleetcd "k8s.io/kubernetes/pkg/registry/rbac/clusterrole/etcd"
	clusterrolepolicybased "k8s.io/kubernetes/pkg/registry/rbac/clusterrole/policybased"
	"k8s.io/kubernetes/pkg/registry/rbac/clusterrolebinding"
	clusterrolebindingetcd "k8s.io/kubernetes/pkg/registry/rbac/clusterrolebinding/etcd"
	clusterrolebindingpolicybased "k8s.io/kubernetes/pkg/registry/rbac/clusterrolebinding/policybased"
	"k8s.io/kubernetes/pkg/registry/rbac/role"
	roleetcd "k8s.io/kubernetes/pkg/registry/rbac/role/etcd"
	rolepolicybased "k8s.io/kubernetes/pkg/registry/rbac/role/policybased"
	"k8s.io/kubernetes/pkg/registry/rbac/rolebinding"
	rolebindingetcd "k8s.io/kubernetes/pkg/registry/rbac/rolebinding/etcd"
	rolebindingpolicybased "k8s.io/kubernetes/pkg/registry/rbac/rolebinding/policybased"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac/bootstrappolicy"
)

type RESTStorageProvider struct {
	AuthorizerRBACSuperUser string
}

var _ genericapiserver.RESTStorageProvider = RESTStorageProvider{}
var _ genericapiserver.PostStartHookProvider = RESTStorageProvider{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter genericapiserver.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(rbac.GroupName)

	if apiResourceConfigSource.AnyResourcesForVersionEnabled(rbacapiv1alpha1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[rbacapiv1alpha1.SchemeGroupVersion.Version] = p.v1alpha1Storage(apiResourceConfigSource, restOptionsGetter)
		apiGroupInfo.GroupMeta.GroupVersion = rbacapiv1alpha1.SchemeGroupVersion
	}

	return apiGroupInfo, true
}

func (p RESTStorageProvider) v1alpha1Storage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter genericapiserver.RESTOptionsGetter) map[string]rest.Storage {
	version := rbacapiv1alpha1.SchemeGroupVersion

	once := new(sync.Once)
	var authorizationRuleResolver rbacvalidation.AuthorizationRuleResolver
	newRuleValidator := func() rbacvalidation.AuthorizationRuleResolver {
		once.Do(func() {
			authorizationRuleResolver = rbacvalidation.NewDefaultRuleResolver(
				role.AuthorizerAdapter{Registry: role.NewRegistry(roleetcd.NewREST(restOptionsGetter(rbac.Resource("roles"))))},
				rolebinding.AuthorizerAdapter{Registry: rolebinding.NewRegistry(rolebindingetcd.NewREST(restOptionsGetter(rbac.Resource("rolebindings"))))},
				clusterrole.AuthorizerAdapter{Registry: clusterrole.NewRegistry(clusterroleetcd.NewREST(restOptionsGetter(rbac.Resource("clusterroles"))))},
				clusterrolebinding.AuthorizerAdapter{Registry: clusterrolebinding.NewRegistry(clusterrolebindingetcd.NewREST(restOptionsGetter(rbac.Resource("clusterrolebindings"))))},
			)
		})
		return authorizationRuleResolver
	}

	storage := map[string]rest.Storage{}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("roles")) {
		rolesStorage := roleetcd.NewREST(restOptionsGetter(rbac.Resource("roles")))
		storage["roles"] = rolepolicybased.NewStorage(rolesStorage, newRuleValidator(), p.AuthorizerRBACSuperUser)
	}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("rolebindings")) {
		roleBindingsStorage := rolebindingetcd.NewREST(restOptionsGetter(rbac.Resource("rolebindings")))
		storage["rolebindings"] = rolebindingpolicybased.NewStorage(roleBindingsStorage, newRuleValidator(), p.AuthorizerRBACSuperUser)
	}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("clusterroles")) {
		clusterRolesStorage := clusterroleetcd.NewREST(restOptionsGetter(rbac.Resource("clusterroles")))
		storage["clusterroles"] = clusterrolepolicybased.NewStorage(clusterRolesStorage, newRuleValidator(), p.AuthorizerRBACSuperUser)
	}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("clusterrolebindings")) {
		clusterRoleBindingsStorage := clusterrolebindingetcd.NewREST(restOptionsGetter(rbac.Resource("clusterrolebindings")))
		storage["clusterrolebindings"] = clusterrolebindingpolicybased.NewStorage(clusterRoleBindingsStorage, newRuleValidator(), p.AuthorizerRBACSuperUser)
	}
	return storage
}

func (p RESTStorageProvider) PostStartHook() (string, genericapiserver.PostStartHookFunc, error) {
	return "rbac/bootstrap-roles", PostStartHook, nil
}

func PostStartHook(hookContext genericapiserver.PostStartHookContext) error {
	clientset, err := rbacclient.NewForConfig(hookContext.LoopbackClientConfig)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to initialize clusterroles: %v", err))
		return nil
	}

	existingClusterRoles, err := clientset.ClusterRoles().List(api.ListOptions{})
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to initialize clusterroles: %v", err))
		return nil
	}
	// if clusterroles already exist, then assume we don't have work to do because we've already
	// initialized or another API server has started this task
	if len(existingClusterRoles.Items) > 0 {
		return nil
	}

	for _, clusterRole := range append(bootstrappolicy.ClusterRoles(), bootstrappolicy.ControllerRoles()...) {
		if _, err := clientset.ClusterRoles().Create(&clusterRole); err != nil {
			// don't fail on failures, try to create as many as you can
			utilruntime.HandleError(fmt.Errorf("unable to initialize clusterroles: %v", err))
			continue
		}
		glog.Infof("Created clusterrole.%s/%s", rbac.GroupName, clusterRole.Name)
	}

	existingClusterRoleBindings, err := clientset.ClusterRoleBindings().List(api.ListOptions{})
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to initialize clusterrolebindings: %v", err))
		return nil
	}
	// if clusterrolebindings already exist, then assume we don't have work to do because we've already
	// initialized or another API server has started this task
	if len(existingClusterRoleBindings.Items) > 0 {
		return nil
	}

	for _, clusterRoleBinding := range append(bootstrappolicy.ClusterRoleBindings(), bootstrappolicy.ControllerRoleBindings()...) {
		if _, err := clientset.ClusterRoleBindings().Create(&clusterRoleBinding); err != nil {
			// don't fail on failures, try to create as many as you can
			utilruntime.HandleError(fmt.Errorf("unable to initialize clusterrolebindings: %v", err))
			continue
		}
		glog.Infof("Created clusterrolebinding.%s/%s", rbac.GroupName, clusterRoleBinding.Name)
	}

	return nil
}

func (p RESTStorageProvider) GroupName() string {
	return rbac.GroupName
}
