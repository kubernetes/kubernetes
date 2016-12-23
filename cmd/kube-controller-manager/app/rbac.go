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

// Package app implements a server that runs a set of active
// components.  This includes replication controllers, service endpoints and
// nodes.
//
package app

import (
	rbacapi "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1"
	internalclientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller/informers"
	clusterrolecontroller "k8s.io/kubernetes/pkg/controller/rbac/clusterrole"
	clusterrolebindingcontroller "k8s.io/kubernetes/pkg/controller/rbac/clusterrolebinding"
	rolecontroller "k8s.io/kubernetes/pkg/controller/rbac/role"
	rolebindingcontroller "k8s.io/kubernetes/pkg/controller/rbac/rolebinding"
	"k8s.io/kubernetes/pkg/runtime/schema"
)

func startRoleController(ctx ControllerContext) (bool, error) {
	if !ctx.AvailableResources[schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1alpha1", Resource: "roles"}] {
		return false, nil
	}
	roleConfig := ctx.ClientBuilder.ConfigOrDie("roles-controller")
	roleConfig.ContentConfig.GroupVersion = &schema.GroupVersion{Group: rbacapi.GroupName, Version: "v1alpha1"}

	informer := informers.NewSharedInformerFactory(nil, internalclientset.NewForConfigOrDie(roleConfig), ResyncPeriod(&ctx.Options)())
	go rolecontroller.NewRoleController(informer.Roles(), internalclientset.NewForConfigOrDie(roleConfig)).Run(ctx.Stop)
	informer.Start(ctx.Stop)
	return true, nil
}

func startRoleBindingController(ctx ControllerContext) (bool, error) {
	if !ctx.AvailableResources[schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1alpha1", Resource: "rolebindings"}] {
		return false, nil
	}
	roleConfig := ctx.ClientBuilder.ConfigOrDie("rolebindings-controller")
	roleConfig.ContentConfig.GroupVersion = &schema.GroupVersion{Group: rbacapi.GroupName, Version: "v1alpha1"}

	informer := informers.NewSharedInformerFactory(nil, internalclientset.NewForConfigOrDie(roleConfig), ResyncPeriod(&ctx.Options)())
	go rolebindingcontroller.NewRoleBindingController(informer.RoleBindings(), internalclientset.NewForConfigOrDie(roleConfig)).Run(ctx.Stop)
	informer.Start(ctx.Stop)
	return true, nil
}

func startClusterRoleController(ctx ControllerContext) (bool, error) {
	if !ctx.AvailableResources[schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1alpha1", Resource: "clusterroles"}] {
		return false, nil
	}
	roleConfig := ctx.ClientBuilder.ConfigOrDie("clusterroles-controller")
	roleConfig.ContentConfig.GroupVersion = &schema.GroupVersion{Group: rbacapi.GroupName, Version: "v1alpha1"}

	informer := informers.NewSharedInformerFactory(nil, internalclientset.NewForConfigOrDie(roleConfig), ResyncPeriod(&ctx.Options)())
	go clusterrolecontroller.NewClusterRoleController(informer.ClusterRoles(), internalclientset.NewForConfigOrDie(roleConfig)).Run(ctx.Stop)
	informer.Start(ctx.Stop)
	return true, nil
}

func startClusterRoleBindingController(ctx ControllerContext) (bool, error) {
	if !ctx.AvailableResources[schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1alpha1", Resource: "clusterrolebindings"}] {
		return false, nil
	}
	roleConfig := ctx.ClientBuilder.ConfigOrDie("clusterrolebindings-controller")
	roleConfig.ContentConfig.GroupVersion = &schema.GroupVersion{Group: rbacapi.GroupName, Version: "v1alpha1"}

	informer := informers.NewSharedInformerFactory(nil, internalclientset.NewForConfigOrDie(roleConfig), ResyncPeriod(&ctx.Options)())
	go clusterrolebindingcontroller.NewClusterRoleBindingController(informer.ClusterRoleBindings(), internalclientset.NewForConfigOrDie(roleConfig)).Run(ctx.Stop)
	informer.Start(ctx.Stop)
	return true, nil
}
