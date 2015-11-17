/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package wmftoolsenforcer

import (
	"io"
	"strconv"
	"time"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

func init() {
	admission.RegisterPlugin("UidEnforcer", func(client client.Interface, config io.Reader) (admission.Interface, error) {
		return NewUidEnforcer(client), nil
	})
}

// plugin contains the client used by the uidenforcer admin controller
type plugin struct {
	*admission.Handler
	client client.Interface
	store  cache.Store
}

// NewUidEnforcer creates a new instance of the UidEnforcer admission controller
func NewUidEnforcer(client client.Interface) admission.Interface {
	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	reflector := cache.NewReflector(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return client.Namespaces().List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return client.Namespaces().Watch(labels.Everything(), fields.Everything(), options)
			},
		},
		&api.Namespace{},
		store,
		5*time.Minute,
	)
	reflector.Run()
	return &plugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
		client:  client,
		store:   store,
	}
}

// This will verify the following:
//  - User object has a numeric uid
//  - Namespace object has an annotation called RunAsUser that's an integer
//
// If after all this there's no SecurityContext on each Container with a RunAsUser set to the same RunAsUser, it'll be set
func (p *plugin) Admit(a admission.Attributes) (err error) {
	if a.GetResource() != string(api.ResourcePods) {
		return nil
	}

	pod, ok := a.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	namespaceObj, exists, err := p.store.Get(&api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:      a.GetNamespace(),
			Namespace: "",
		},
	})

	if !exists {
		return apierrors.NewBadRequest("Namespace " + a.GetNamespace() + " not found")
	}

	if err != nil {
		return apierrors.NewBadRequest("Everything must be in a namespace!")
	}
	namespace := namespaceObj.(*api.Namespace)

	if _, ok := namespace.Annotations["RunAsUser"]; !ok {
		return apierrors.NewBadRequest("Namespace does not have a RunAsUser annotation!")
	}

	for i := 0; i < len(pod.Spec.Containers); i++ {
		container := &pod.Spec.Containers[i]
		uid, ok := strconv.ParseInt(namespace.Annotations["RunAsUser"], 10, 32)
		if ok == nil {
			if container.SecurityContext == nil {
				container.SecurityContext = &api.SecurityContext{
					RunAsUser: &uid,
				}
			} else {
				return apierrors.NewBadRequest("Must have an empty SecuriyContext to pass!")
			}
		} else {
			return apierrors.NewBadRequest("Namespace's RunAsUser not an integer")
		}
	}
	return nil
}
