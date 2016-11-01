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

package informers

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/client/cache"
)

// GenericInformer is type of SharedIndexInformer which will locate and delegate to other
// sharedInformers based on type
type GenericInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() cache.GenericLister
}

// ForResource gives generic access to a shared informer of the matching type
// TODO extend this to unknown resources with a client pool
func (f *sharedInformerFactory) ForResource(resource unversioned.GroupResource) (GenericInformer, error) {
	switch resource {
	case api.Resource("pods"):
		return &genericInformer{resource: resource, informer: f.Pods().Informer()}, nil
	case api.Resource("limitranges"):
		return &genericInformer{resource: resource, informer: f.LimitRanges().Informer()}, nil
	case api.Resource("namespaces"):
		return &genericInformer{resource: resource, informer: f.Namespaces().Informer()}, nil
	case api.Resource("nodes"):
		return &genericInformer{resource: resource, informer: f.Nodes().Informer()}, nil
	case api.Resource("persistentvolumeclaims"):
		return &genericInformer{resource: resource, informer: f.PersistentVolumeClaims().Informer()}, nil
	case api.Resource("persistentvolumes"):
		return &genericInformer{resource: resource, informer: f.PersistentVolumes().Informer()}, nil
	case api.Resource("serviceaccounts"):
		return &genericInformer{resource: resource, informer: f.ServiceAccounts().Informer()}, nil

	case extensions.Resource("daemonsets"):
		return &genericInformer{resource: resource, informer: f.DaemonSets().Informer()}, nil
	case extensions.Resource("deployments"):
		return &genericInformer{resource: resource, informer: f.Deployments().Informer()}, nil
	case extensions.Resource("replicasets"):
		return &genericInformer{resource: resource, informer: f.ReplicaSets().Informer()}, nil

	case rbac.Resource("clusterrolebindings"):
		return &genericInformer{resource: resource, informer: f.ClusterRoleBindings().Informer()}, nil
	case rbac.Resource("clusterroles"):
		return &genericInformer{resource: resource, informer: f.ClusterRoles().Informer()}, nil
	case rbac.Resource("rolebindings"):
		return &genericInformer{resource: resource, informer: f.RoleBindings().Informer()}, nil
	case rbac.Resource("roles"):
		return &genericInformer{resource: resource, informer: f.Roles().Informer()}, nil

	case batch.Resource("jobs"):
		return &genericInformer{resource: resource, informer: f.Jobs().Informer()}, nil
	}

	return nil, fmt.Errorf("no informer found for %v", resource)
}

type genericInformer struct {
	informer cache.SharedIndexInformer
	resource unversioned.GroupResource
}

func (f *genericInformer) Informer() cache.SharedIndexInformer {
	return f.informer
}

func (f *genericInformer) Lister() cache.GenericLister {
	return cache.NewGenericLister(f.Informer().GetIndexer(), f.resource)
}
