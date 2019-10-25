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

package rest

import (
	"fmt"
	"time"

	"k8s.io/klog"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	clientset "k8s.io/client-go/kubernetes"
	schedulingclient "k8s.io/client-go/kubernetes/typed/scheduling/v1"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	schedulingapiv1 "k8s.io/kubernetes/pkg/apis/scheduling/v1"
	schedulingapiv1alpha1 "k8s.io/kubernetes/pkg/apis/scheduling/v1alpha1"
	schedulingapiv1beta1 "k8s.io/kubernetes/pkg/apis/scheduling/v1beta1"
	priorityclassstore "k8s.io/kubernetes/pkg/registry/scheduling/priorityclass/storage"
)

const (
	PostStartHookName = "scheduling/bootstrap-scheduler-defaults"

	// DefaultSystemQuotaName is the name of default system quota which allows unlimited
	// system critical pods to be created
	DefaultSystemQuotaName = "default-system-quota"
)

type RESTStorageProvider struct{}

var _ genericapiserver.PostStartHookProvider = RESTStorageProvider{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool, error) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(scheduling.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)

	if apiResourceConfigSource.VersionEnabled(schedulingapiv1alpha1.SchemeGroupVersion) {
		if storage, err := p.v1alpha1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		} else {
			apiGroupInfo.VersionedResourcesStorageMap[schedulingapiv1alpha1.SchemeGroupVersion.Version] = storage
		}
	}
	if apiResourceConfigSource.VersionEnabled(schedulingapiv1beta1.SchemeGroupVersion) {
		if storage, err := p.v1beta1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		} else {
			apiGroupInfo.VersionedResourcesStorageMap[schedulingapiv1beta1.SchemeGroupVersion.Version] = storage
		}
	}
	if apiResourceConfigSource.VersionEnabled(schedulingapiv1.SchemeGroupVersion) {
		if storage, err := p.v1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		} else {
			apiGroupInfo.VersionedResourcesStorageMap[schedulingapiv1.SchemeGroupVersion.Version] = storage
		}
	}
	return apiGroupInfo, true, nil
}

func (p RESTStorageProvider) v1alpha1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}
	// priorityclasses
	if priorityClassStorage, err := priorityclassstore.NewREST(restOptionsGetter); err != nil {
		return nil, err
	} else {
		storage["priorityclasses"] = priorityClassStorage
	}
	return storage, nil
}

func (p RESTStorageProvider) v1beta1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}
	// priorityclasses
	if priorityClassStorage, err := priorityclassstore.NewREST(restOptionsGetter); err != nil {
		return nil, err
	} else {
		storage["priorityclasses"] = priorityClassStorage
	}

	return storage, nil
}

func (p RESTStorageProvider) v1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}
	// priorityclasses
	if priorityClassStorage, err := priorityclassstore.NewREST(restOptionsGetter); err != nil {
		return nil, err
	} else {
		storage["priorityclasses"] = priorityClassStorage
	}

	return storage, nil
}

func (p RESTStorageProvider) PostStartHook() (string, genericapiserver.PostStartHookFunc, error) {
	return PostStartHookName, addSchedulingDefaults(), nil
}

func (p RESTStorageProvider) GroupName() string {
	return scheduling.GroupName
}

// addSchedulingDefaults adds the default cluster critical priorityClasses and clusterResourceQuota which allows
// unlimited number of critical pods be created in `kube-system` namespace.
func addSchedulingDefaults() genericapiserver.PostStartHookFunc {
	return func(hookContext genericapiserver.PostStartHookContext) error {
		// Adding system priority classes is important. If they fail to add, many critical system
		// components may fail and cluster may break.
		if err := addSystemPriorityClasses(hookContext); err != nil {
			return fmt.Errorf("unable to add default system priority classes: %v", err)
		}
		// Add default resource quota which allows unlimited number of critical pods to be created in kube-system
		// namespace. This is to ensure backwards compatibility with existing system where we limit critical pods
		// to be created in `kube-system`.
		if err := addDefaultSystemQuota(hookContext); err != nil {
			return fmt.Errorf("unable to add default system resource quota: %v", err)
		}
		return nil
	}
}

func addSystemPriorityClasses(hookContext genericapiserver.PostStartHookContext) error {
	return wait.Poll(1*time.Second, 30*time.Second, func() (done bool, err error) {
		schedClientSet, err := schedulingclient.NewForConfig(hookContext.LoopbackClientConfig)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("unable to initialize client: %v", err))
			return false, nil
		}

		for _, pc := range schedulingapiv1.SystemPriorityClasses() {
			_, err := schedClientSet.PriorityClasses().Get(pc.Name, metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				_, err := schedClientSet.PriorityClasses().Create(pc)
				if err != nil && !apierrors.IsAlreadyExists(err) {
					return false, err
				}

				klog.Infof("created PriorityClass %s with value %v", pc.Name, pc.Value)
			}

			if err != nil {
				// Unable to get the priority class for reasons other than "not found".
				klog.Warningf("unable to get PriorityClass %v: %v. Retrying...", pc.Name, err)
				return false, nil
			}
		}
		klog.Infof("all system priority classes are created successfully or already exist.")
		return true, nil
	})
}

// addDefaultSystemQuota adds a quota which allows unlimited critical pods to be created in `kube-system` namespace
// This quota can be deleted/updated by cluster-admin. If the cluster-admin wants critical pods to be created in
// namespace other than `kube-system`, he/she can configure quotas which allows these critical pods to be created
// in that namespace.
func addDefaultSystemQuota(hookContext genericapiserver.PostStartHookContext) error {
	return wait.Poll(1*time.Second, 30*time.Second, func() (done bool, err error) {
		client := clientset.NewForConfigOrDie(hookContext.LoopbackClientConfig)
		_, err = client.CoreV1().ResourceQuotas(metav1.NamespaceSystem).Get(DefaultSystemQuotaName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			_, err := client.CoreV1().ResourceQuotas(metav1.NamespaceSystem).Create(systemDefaultQuota())
			if err != nil && !apierrors.IsAlreadyExists(err) {
				klog.V(5).Infof("Unable to create system default resourceQuota %v: %v. Retrying..", DefaultSystemQuotaName, err)
				return false, nil
			}

			klog.V(5).Infof("Created defaultResourceQuota %v which allows unlimited critical pods to be created in kube-system namespace", DefaultSystemQuotaName)
		}

		if err != nil {
			// Unable to get the default resource quota for reasons other than "not found".
			klog.Warningf("Unable to get system default resourceQuota %v: %v. Retrying..", DefaultSystemQuotaName, err)
			return false, nil
		}
		klog.Info("Required default resource quota is created or already exist")
		return true, nil
	})
}

// systemDefaultQuota returns a default system quota. This default quota allows unlimited number of critical pods
// to be created in `kube-system` namespace. This is needed for backwards compatibility with current system
// as we allow unlimited critical pods to be created in `kube-system` namespace. The DefaultSystemQuotaName gets
// automatically created when we start kube-apiserver. The cluster-admin is allowed to create/delete or update
// this quota. In order to create critical pods in other namespaces, cluster-admin can create quotas
// in those namespaces.
func systemDefaultQuota() *v1.ResourceQuota {
	defaultQuota := &v1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: DefaultSystemQuotaName, Namespace: metav1.NamespaceSystem},
		Spec: v1.ResourceQuotaSpec{
			ScopeSelector: &v1.ScopeSelector{
				MatchExpressions: []v1.ScopedResourceSelectorRequirement{
					{
						ScopeName: v1.ResourceQuotaScopePriorityClass,
						Operator:  v1.ScopeSelectorOpIn,
						Values:    []string{"system-cluster-critical", "system-node-critical"},
					},
				},
			},
		},
	}
	return defaultQuota
}
