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

package core

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/generic"
)

// NewRegistry returns a registry that knows how to deal with core kubernetes resources
// If an informer factory is provided, evaluators will use them.
func NewRegistry(kubeClient clientset.Interface, f informers.SharedInformerFactory) quota.Registry {
	pod := NewPodEvaluator(kubeClient, f)
	service := NewServiceEvaluator(kubeClient)
	replicationController := NewReplicationControllerEvaluator(kubeClient)
	resourceQuota := NewResourceQuotaEvaluator(kubeClient)
	secret := NewSecretEvaluator(kubeClient)
	configMap := NewConfigMapEvaluator(kubeClient)
	persistentVolumeClaim := NewPersistentVolumeClaimEvaluator(kubeClient, f)
	return &generic.GenericRegistry{
		InternalEvaluators: map[schema.GroupKind]quota.Evaluator{
			pod.GroupKind():                   pod,
			service.GroupKind():               service,
			replicationController.GroupKind(): replicationController,
			secret.GroupKind():                secret,
			configMap.GroupKind():             configMap,
			resourceQuota.GroupKind():         resourceQuota,
			persistentVolumeClaim.GroupKind(): persistentVolumeClaim,
		},
	}
}
