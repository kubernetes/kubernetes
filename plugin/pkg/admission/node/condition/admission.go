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

// Package condition contains a NodeCondition admission controller that sets Node Conditions on new Node.
// This is useful because Node.Status.Conditions can be used to decouple our controllers, so that a new node
// will not be ready until various processes have had the chance to run on it.
// We want to ensure that a new node has a list of unhealthy Status Conditions set on creation, but we
// can't hard-code a list of conditions into the kubelet, or else we have coupled things again.
// Enter the NodeCondition admission controller: it will apply Status conditions on Node creation.
package condition

import (
	"flag"
	"io"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/util/sets"
)

var (
	flagConditions = flag.String("new-node-conditions", "", "Comma-separated list of conditions to apply on newly created Nodes.")
)

const ReasonPending = "Pending"

func init() {
	admission.RegisterPlugin("NodeCondition", func(client clientset.Interface, config io.Reader) (admission.Interface, error) {
		conditions := sets.NewString()
		for _, s := range strings.Split(*flagConditions, ",") {
			s = strings.TrimSpace(s)
			if s != "" {
				conditions.Insert(s)
			}
		}
		if conditions.Len() == 0 {
			return nil, nil
		}
		return newNodeCondition(conditions.List()), nil
	})
}

// nodeCondition is an implementation of admission.Interface.
// It looks at all new Nodes and sets Conditions.
type nodeCondition struct {
	*admission.Handler
	conditions []string
}

func (a *nodeCondition) Admit(attributes admission.Attributes) (err error) {
	// Ignore all calls to subresources or resources other than Nodes.
	if len(attributes.GetSubresource()) != 0 || attributes.GetResource().GroupResource() != api.Resource("nodes") {
		return nil
	}
	node, ok := attributes.GetObject().(*api.Node)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Node but was unable to be converted")
	}

	preexisting := sets.NewString()
	for i := range node.Status.Conditions {
		preexisting.Insert(string(node.Status.Conditions[i].Type))
	}

	for _, c := range a.conditions {
		if preexisting.Has(c) {
			continue
		}

		node.Status.Conditions = append(node.Status.Conditions, api.NodeCondition{
			Type:               api.NodeConditionType(c),
			Status:             api.ConditionTrue,
			Reason:             ReasonPending,
			Message:            "Condition automatically set on node creation",
			LastTransitionTime: unversioned.NewTime(time.Now()),
		})
	}

	return nil
}

// newNodeCondition creates a new NodeCondition admission control handler
func newNodeCondition(conditions []string) admission.Interface {
	return &nodeCondition{
		Handler:    admission.NewHandler(admission.Create),
		conditions: conditions,
	}
}
