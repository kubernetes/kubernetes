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

package dedicatedmachines

import (
	"io"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

func init() {
	admission.RegisterPlugin("DedicatedMachines", func(client client.Interface, config io.Reader) (admission.Interface, error) {
		return NewDedicated(client), nil
	})
}

// provision is an implementation of admission.Interface.
// It looks at all incoming requests in a namespace context, and if the namespace does not exist, it creates one.
// It is useful in deployments that do not want to restrict creation of a namespace prior to its usage.
type dedicated struct {
	*admission.Handler
	client          client.Interface
	extensionClient client.ExtensionsInterface
	store           cache.Store
}

func (d *dedicated) Admit(a admission.Attributes) (err error) {
	// Ignore all calls to subresources or resources other than pods.
	if a.GetSubresource() != "" || a.GetResource() != string(api.ResourcePods) {
		return nil
	}
	pod, ok := a.GetObject().(*api.Pod)
	if !ok {
		return errors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	dedicatedMachines, err := d.extensionClient.DedicatedMachines(a.GetNamespace()).List(labels.Everything(), fields.Everything(), unversioned.ListOptions{})
	if err != nil {
		return admission.NewForbidden(a, err)
	}
	if len(dedicatedMachines.Items) == 0 {
		return nil
	}

	// TODO: should use setter instead of setting podSpec attributes directly here?
	if pod.Spec.NodeSelector == nil {
		pod.Spec.NodeSelector = map[string]string{}
	}

	// TODO: Currently only support one dedicatedMachine's LabelValue,
	// when nodeSelector supports 'IN' operator, this need to be improved.
	pod.Spec.NodeSelector["dedicated"] = dedicatedMachines.Items[0].Spec.LabelValue
	return nil
}

// NewProvision creates a new namespace provision admission control handler
func NewDedicated(c client.Interface) admission.Interface {
	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	reflector := cache.NewReflector(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return c.Namespaces().List(labels.Everything(), fields.Everything(), unversioned.ListOptions{})
			},
			WatchFunc: func(options unversioned.ListOptions) (watch.Interface, error) {
				return c.Namespaces().Watch(options)
			},
		},
		&api.Namespace{},
		store,
		0,
	)
	reflector.Run()
	return &dedicated{
		client:  c,
		store:   store,
		Handler: admission.NewHandler(admission.Create, admission.Update, admission.Delete),
	}
}

func createProvision(c client.Interface, store cache.Store) admission.Interface {
	return &dedicated{
		Handler: admission.NewHandler(admission.Create),
		client:  c,
		store:   store,
	}
}
