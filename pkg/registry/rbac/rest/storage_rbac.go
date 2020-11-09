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
	"context"
	"fmt"
	"time"

	"k8s.io/klog/v2"

	rbacapiv1 "k8s.io/api/rbac/v1"
	rbacapiv1alpha1 "k8s.io/api/rbac/v1alpha1"
	rbacapiv1beta1 "k8s.io/api/rbac/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	rbacv1client "k8s.io/client-go/kubernetes/typed/rbac/v1"
	"k8s.io/client-go/util/retry"
	"k8s.io/component-helpers/auth/rbac/reconciliation"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/rbac"
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

const PostStartHookName = "rbac/bootstrap-roles"

type RESTStorageProvider struct {
	Authorizer authorizer.Authorizer
}

var _ genericapiserver.PostStartHookProvider = RESTStorageProvider{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool, error) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(rbac.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)
	// If you add a version here, be sure to add an entry in `k8s.io/kubernetes/cmd/kube-apiserver/app/aggregator.go with specific priorities.
	// TODO refactor the plumbing to provide the information in the APIGroupInfo

	if apiResourceConfigSource.VersionEnabled(rbacapiv1alpha1.SchemeGroupVersion) {
		if storageMap, err := p.storage(rbacapiv1alpha1.SchemeGroupVersion, apiResourceConfigSource, restOptionsGetter); err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		} else {
			apiGroupInfo.VersionedResourcesStorageMap[rbacapiv1alpha1.SchemeGroupVersion.Version] = storageMap
		}
	}
	if apiResourceConfigSource.VersionEnabled(rbacapiv1beta1.SchemeGroupVersion) {
		if storageMap, err := p.storage(rbacapiv1beta1.SchemeGroupVersion, apiResourceConfigSource, restOptionsGetter); err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		} else {
			apiGroupInfo.VersionedResourcesStorageMap[rbacapiv1beta1.SchemeGroupVersion.Version] = storageMap
		}
	}
	if apiResourceConfigSource.VersionEnabled(rbacapiv1.SchemeGroupVersion) {
		if storageMap, err := p.storage(rbacapiv1.SchemeGroupVersion, apiResourceConfigSource, restOptionsGetter); err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		} else {
			apiGroupInfo.VersionedResourcesStorageMap[rbacapiv1.SchemeGroupVersion.Version] = storageMap
		}
	}

	return apiGroupInfo, true, nil
}

func (p RESTStorageProvider) storage(version schema.GroupVersion, apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}
	rolesStorage, err := rolestore.NewREST(restOptionsGetter)
	if err != nil {
		return storage, err
	}
	roleBindingsStorage, err := rolebindingstore.NewREST(restOptionsGetter)
	if err != nil {
		return storage, err
	}
	clusterRolesStorage, err := clusterrolestore.NewREST(restOptionsGetter)
	if err != nil {
		return storage, err
	}
	clusterRoleBindingsStorage, err := clusterrolebindingstore.NewREST(restOptionsGetter)
	if err != nil {
		return storage, err
	}

	authorizationRuleResolver := rbacregistryvalidation.NewDefaultRuleResolver(
		role.AuthorizerAdapter{Registry: role.NewRegistry(rolesStorage)},
		rolebinding.AuthorizerAdapter{Registry: rolebinding.NewRegistry(roleBindingsStorage)},
		clusterrole.AuthorizerAdapter{Registry: clusterrole.NewRegistry(clusterRolesStorage)},
		clusterrolebinding.AuthorizerAdapter{Registry: clusterrolebinding.NewRegistry(clusterRoleBindingsStorage)},
	)

	// roles
	storage["roles"] = rolepolicybased.NewStorage(rolesStorage, p.Authorizer, authorizationRuleResolver)

	// rolebindings
	storage["rolebindings"] = rolebindingpolicybased.NewStorage(roleBindingsStorage, p.Authorizer, authorizationRuleResolver)

	// clusterroles
	storage["clusterroles"] = clusterrolepolicybased.NewStorage(clusterRolesStorage, p.Authorizer, authorizationRuleResolver)

	// clusterrolebindings
	storage["clusterrolebindings"] = clusterrolebindingpolicybased.NewStorage(clusterRoleBindingsStorage, p.Authorizer, authorizationRuleResolver)

	return storage, nil
}

func (p RESTStorageProvider) PostStartHook() (string, genericapiserver.PostStartHookFunc, error) {
	policy := &PolicyData{
		ClusterRoles:               append(bootstrappolicy.ClusterRoles(), bootstrappolicy.ControllerRoles()...),
		ClusterRoleBindings:        append(bootstrappolicy.ClusterRoleBindings(), bootstrappolicy.ControllerRoleBindings()...),
		Roles:                      bootstrappolicy.NamespaceRoles(),
		RoleBindings:               bootstrappolicy.NamespaceRoleBindings(),
		ClusterRolesToAggregate:    bootstrappolicy.ClusterRolesToAggregate(),
		ClusterRoleBindingsToSplit: bootstrappolicy.ClusterRoleBindingsToSplit(),
	}
	return PostStartHookName, policy.EnsureRBACPolicy(), nil
}

type PolicyData struct {
	ClusterRoles        []rbacapiv1.ClusterRole
	ClusterRoleBindings []rbacapiv1.ClusterRoleBinding
	Roles               map[string][]rbacapiv1.Role
	RoleBindings        map[string][]rbacapiv1.RoleBinding
	// ClusterRolesToAggregate maps from previous clusterrole name to the new clusterrole name
	ClusterRolesToAggregate map[string]string
	// ClusterRoleBindingsToSplit maps from previous ClusterRoleBinding Name to a template for the new ClusterRoleBinding
	ClusterRoleBindingsToSplit map[string]rbacapiv1.ClusterRoleBinding
}

func (p *PolicyData) EnsureRBACPolicy() genericapiserver.PostStartHookFunc {
	return func(hookContext genericapiserver.PostStartHookContext) error {
		// initializing roles is really important.  On some e2e runs, we've seen cases where etcd is down when the server
		// starts, the roles don't initialize, and nothing works.
		err := wait.Poll(1*time.Second, 30*time.Second, func() (done bool, err error) {

			coreclientset, err := corev1client.NewForConfig(hookContext.LoopbackClientConfig)
			if err != nil {
				utilruntime.HandleError(fmt.Errorf("unable to initialize client: %v", err))
				return false, nil
			}

			clientset, err := rbacv1client.NewForConfig(hookContext.LoopbackClientConfig)
			if err != nil {
				utilruntime.HandleError(fmt.Errorf("unable to initialize client: %v", err))
				return false, nil
			}
			// Make sure etcd is responding before we start reconciling
			if _, err := clientset.ClusterRoles().List(context.TODO(), metav1.ListOptions{}); err != nil {
				utilruntime.HandleError(fmt.Errorf("unable to initialize clusterroles: %v", err))
				return false, nil
			}
			if _, err := clientset.ClusterRoleBindings().List(context.TODO(), metav1.ListOptions{}); err != nil {
				utilruntime.HandleError(fmt.Errorf("unable to initialize clusterrolebindings: %v", err))
				return false, nil
			}

			// if the new cluster roles to aggregate do not yet exist, then we need to copy the old roles if they don't exist
			// in new locations
			if err := primeAggregatedClusterRoles(p.ClusterRolesToAggregate, clientset); err != nil {
				utilruntime.HandleError(fmt.Errorf("unable to prime aggregated clusterroles: %v", err))
				return false, nil
			}

			if err := primeSplitClusterRoleBindings(p.ClusterRoleBindingsToSplit, clientset); err != nil {
				utilruntime.HandleError(fmt.Errorf("unable to prime split ClusterRoleBindings: %v", err))
				return false, nil
			}

			// ensure bootstrap roles are created or reconciled
			for _, clusterRole := range p.ClusterRoles {
				opts := reconciliation.ReconcileRoleOptions{
					Role:    reconciliation.ClusterRoleRuleOwner{ClusterRole: &clusterRole},
					Client:  reconciliation.ClusterRoleModifier{Client: clientset.ClusterRoles()},
					Confirm: true,
				}
				err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
					result, err := opts.Run()
					if err != nil {
						return err
					}
					switch {
					case result.Protected && result.Operation != reconciliation.ReconcileNone:
						klog.Warningf("skipped reconcile-protected clusterrole.%s/%s with missing permissions: %v", rbac.GroupName, clusterRole.Name, result.MissingRules)
					case result.Operation == reconciliation.ReconcileUpdate:
						klog.V(2).Infof("updated clusterrole.%s/%s with additional permissions: %v", rbac.GroupName, clusterRole.Name, result.MissingRules)
					case result.Operation == reconciliation.ReconcileCreate:
						klog.V(2).Infof("created clusterrole.%s/%s", rbac.GroupName, clusterRole.Name)
					}
					return nil
				})
				if err != nil {
					// don't fail on failures, try to create as many as you can
					utilruntime.HandleError(fmt.Errorf("unable to reconcile clusterrole.%s/%s: %v", rbac.GroupName, clusterRole.Name, err))
				}
			}

			// ensure bootstrap rolebindings are created or reconciled
			for _, clusterRoleBinding := range p.ClusterRoleBindings {
				opts := reconciliation.ReconcileRoleBindingOptions{
					RoleBinding: reconciliation.ClusterRoleBindingAdapter{ClusterRoleBinding: &clusterRoleBinding},
					Client:      reconciliation.ClusterRoleBindingClientAdapter{Client: clientset.ClusterRoleBindings()},
					Confirm:     true,
				}
				err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
					result, err := opts.Run()
					if err != nil {
						return err
					}
					switch {
					case result.Protected && result.Operation != reconciliation.ReconcileNone:
						klog.Warningf("skipped reconcile-protected clusterrolebinding.%s/%s with missing subjects: %v", rbac.GroupName, clusterRoleBinding.Name, result.MissingSubjects)
					case result.Operation == reconciliation.ReconcileUpdate:
						klog.V(2).Infof("updated clusterrolebinding.%s/%s with additional subjects: %v", rbac.GroupName, clusterRoleBinding.Name, result.MissingSubjects)
					case result.Operation == reconciliation.ReconcileCreate:
						klog.V(2).Infof("created clusterrolebinding.%s/%s", rbac.GroupName, clusterRoleBinding.Name)
					case result.Operation == reconciliation.ReconcileRecreate:
						klog.V(2).Infof("recreated clusterrolebinding.%s/%s", rbac.GroupName, clusterRoleBinding.Name)
					}
					return nil
				})
				if err != nil {
					// don't fail on failures, try to create as many as you can
					utilruntime.HandleError(fmt.Errorf("unable to reconcile clusterrolebinding.%s/%s: %v", rbac.GroupName, clusterRoleBinding.Name, err))
				}
			}

			// ensure bootstrap namespaced roles are created or reconciled
			for namespace, roles := range p.Roles {
				for _, role := range roles {
					opts := reconciliation.ReconcileRoleOptions{
						Role:    reconciliation.RoleRuleOwner{Role: &role},
						Client:  reconciliation.RoleModifier{Client: clientset, NamespaceClient: coreclientset.Namespaces()},
						Confirm: true,
					}
					err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
						result, err := opts.Run()
						if err != nil {
							return err
						}
						switch {
						case result.Protected && result.Operation != reconciliation.ReconcileNone:
							klog.Warningf("skipped reconcile-protected role.%s/%s in %v with missing permissions: %v", rbac.GroupName, role.Name, namespace, result.MissingRules)
						case result.Operation == reconciliation.ReconcileUpdate:
							klog.V(2).Infof("updated role.%s/%s in %v with additional permissions: %v", rbac.GroupName, role.Name, namespace, result.MissingRules)
						case result.Operation == reconciliation.ReconcileCreate:
							klog.V(2).Infof("created role.%s/%s in %v", rbac.GroupName, role.Name, namespace)
						}
						return nil
					})
					if err != nil {
						// don't fail on failures, try to create as many as you can
						utilruntime.HandleError(fmt.Errorf("unable to reconcile role.%s/%s in %v: %v", rbac.GroupName, role.Name, namespace, err))
					}
				}
			}

			// ensure bootstrap namespaced rolebindings are created or reconciled
			for namespace, roleBindings := range p.RoleBindings {
				for _, roleBinding := range roleBindings {
					opts := reconciliation.ReconcileRoleBindingOptions{
						RoleBinding: reconciliation.RoleBindingAdapter{RoleBinding: &roleBinding},
						Client:      reconciliation.RoleBindingClientAdapter{Client: clientset, NamespaceClient: coreclientset.Namespaces()},
						Confirm:     true,
					}
					err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
						result, err := opts.Run()
						if err != nil {
							return err
						}
						switch {
						case result.Protected && result.Operation != reconciliation.ReconcileNone:
							klog.Warningf("skipped reconcile-protected rolebinding.%s/%s in %v with missing subjects: %v", rbac.GroupName, roleBinding.Name, namespace, result.MissingSubjects)
						case result.Operation == reconciliation.ReconcileUpdate:
							klog.V(2).Infof("updated rolebinding.%s/%s in %v with additional subjects: %v", rbac.GroupName, roleBinding.Name, namespace, result.MissingSubjects)
						case result.Operation == reconciliation.ReconcileCreate:
							klog.V(2).Infof("created rolebinding.%s/%s in %v", rbac.GroupName, roleBinding.Name, namespace)
						case result.Operation == reconciliation.ReconcileRecreate:
							klog.V(2).Infof("recreated rolebinding.%s/%s in %v", rbac.GroupName, roleBinding.Name, namespace)
						}
						return nil
					})
					if err != nil {
						// don't fail on failures, try to create as many as you can
						utilruntime.HandleError(fmt.Errorf("unable to reconcile rolebinding.%s/%s in %v: %v", rbac.GroupName, roleBinding.Name, namespace, err))
					}
				}
			}

			return true, nil
		})
		// if we're never able to make it through initialization, kill the API server
		if err != nil {
			return fmt.Errorf("unable to initialize roles: %v", err)
		}

		return nil
	}
}

func (p RESTStorageProvider) GroupName() string {
	return rbac.GroupName
}

// primeAggregatedClusterRoles copies roles that have transitioned to aggregated roles and may need to pick up changes
// that were done to the legacy roles.
func primeAggregatedClusterRoles(clusterRolesToAggregate map[string]string, clusterRoleClient rbacv1client.ClusterRolesGetter) error {
	for oldName, newName := range clusterRolesToAggregate {
		_, err := clusterRoleClient.ClusterRoles().Get(context.TODO(), newName, metav1.GetOptions{})
		if err == nil {
			continue
		}
		if !apierrors.IsNotFound(err) {
			return err
		}

		existingRole, err := clusterRoleClient.ClusterRoles().Get(context.TODO(), oldName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			continue
		}
		if err != nil {
			return err
		}
		if existingRole.AggregationRule != nil {
			// the old role already moved to an aggregated role, so there are no custom rules to migrate at this point
			return nil
		}
		klog.V(1).Infof("migrating %v to %v", existingRole.Name, newName)
		existingRole.Name = newName
		existingRole.ResourceVersion = "" // clear this so the object can be created.
		if _, err := clusterRoleClient.ClusterRoles().Create(context.TODO(), existingRole, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
			return err
		}
	}

	return nil
}

// primeSplitClusterRoleBindings ensures the existence of target ClusterRoleBindings
// by copying Subjects, Annotations, and Labels from the specified source
// ClusterRoleBinding, if present.
func primeSplitClusterRoleBindings(clusterRoleBindingToSplit map[string]rbacapiv1.ClusterRoleBinding, clusterRoleBindingClient rbacv1client.ClusterRoleBindingsGetter) error {
	for existingBindingName, clusterRoleBindingToCreate := range clusterRoleBindingToSplit {
		// If source ClusterRoleBinding does not exist, do nothing.
		existingRoleBinding, err := clusterRoleBindingClient.ClusterRoleBindings().Get(context.TODO(), existingBindingName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			continue
		}
		if err != nil {
			return err
		}

		// If the target ClusterRoleBinding already exists, do nothing.
		_, err = clusterRoleBindingClient.ClusterRoleBindings().Get(context.TODO(), clusterRoleBindingToCreate.Name, metav1.GetOptions{})
		if err == nil {
			continue
		}
		if !apierrors.IsNotFound(err) {
			return err
		}

		// If the source exists, but the target does not,
		// copy the subjects, labels, and annotations from the former to create the latter.
		klog.V(1).Infof("copying subjects, labels, and annotations from ClusterRoleBinding %q to template %q", existingBindingName, clusterRoleBindingToCreate.Name)
		newCRB := clusterRoleBindingToCreate.DeepCopy()
		newCRB.Subjects = existingRoleBinding.Subjects
		newCRB.Labels = existingRoleBinding.Labels
		newCRB.Annotations = existingRoleBinding.Annotations
		if _, err := clusterRoleBindingClient.ClusterRoleBindings().Create(context.TODO(), newCRB, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
			return err
		}
	}
	return nil
}
