/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package node

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"

	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/validation"
)

const (
	labelPrefix = "k8s.mesosphere.io/attribute-"
)

// Create creates a new node api object with the given hostname and labels
func Create(client *client.Client, hostName string, labels map[string]string) (*api.Node, error) {
	n := api.Node{
		ObjectMeta: api.ObjectMeta{
			Name:   hostName,
			Labels: map[string]string{"kubernetes.io/hostname": hostName},
		},
		Spec: api.NodeSpec{
			ExternalID: hostName,
		},
		Status: api.NodeStatus{
			Phase: api.NodePending,
		},
	}
	for k, v := range labels {
		n.Labels[k] = v
	}

	// try to create
	return client.Nodes().Create(&n)
}

// Update updates an existing node api object with new labels
func Update(client *client.Client, n *api.Node, labels map[string]string) (*api.Node, error) {
	patch := struct {
		Metadata struct {
			Labels map[string]string `json:"labels"`
		} `json:"metadata"`
	}{}
	patch.Metadata.Labels = map[string]string{}
	for k, v := range n.Labels {
		if !IsSlaveAttributeLabel(k) {
			patch.Metadata.Labels[k] = v
		}
	}
	for k, v := range labels {
		patch.Metadata.Labels[k] = v
	}
	patchJson, _ := json.Marshal(patch)
	log.V(4).Infof("Patching labels of node %q: %v", n.Name, string(patchJson))
	err := client.Patch(api.MergePatchType).RequestURI(n.SelfLink).Body(patchJson).Do().Error()
	if err != nil {
		return nil, fmt.Errorf("error updating labels of node %q: %v", n.Name, err)
	}

	newNode, err := api.Scheme.DeepCopy(n)
	if err != nil {
		return nil, err
	}
	newNode.(*api.Node).Labels = patch.Metadata.Labels

	return newNode.(*api.Node), nil
}

// CreateOrUpdate tries to create a node api object or updates an already existing one
func CreateOrUpdate(client *client.Client, hostName string, labels map[string]string) (*api.Node, error) {
	n, err := Create(client, hostName, labels)
	if err == nil {
		return n, nil
	}
	if !errors.IsAlreadyExists(err) {
		return nil, fmt.Errorf("unable to register %q with the apiserver: %v", hostName, err)
	}

	// fall back to update an old node with new labels
	n, err = client.Nodes().Get(hostName)
	if err != nil {
		return nil, fmt.Errorf("error getting node %q: %v", hostName, err)
	}
	if n == nil {
		return nil, fmt.Errorf("no node instance returned for %q", hostName)
	}
	return Update(client, n, labels)
}

// IsSlaveAttributeLabel returns true iff the given label is derived from a slave attribute
func IsSlaveAttributeLabel(l string) bool {
	return strings.HasPrefix(l, labelPrefix)
}

// IsUpToDate returns true iff the node's slave labels match the given attributes labels
func IsUpToDate(n *api.Node, labels map[string]string) bool {
	slaveLabels := map[string]string{}
	for k, v := range n.Labels {
		if IsSlaveAttributeLabel(k) {
			slaveLabels[k] = v
		}
	}
	return reflect.DeepEqual(slaveLabels, labels)
}

// SlaveAttributesToLabels converts slave attributes into string key/value labels
func SlaveAttributesToLabels(attrs []*mesos.Attribute) map[string]string {
	l := map[string]string{}
	for _, a := range attrs {
		if a == nil {
			continue
		}

		var v string
		k := labelPrefix + a.GetName()

		switch a.GetType() {
		case mesos.Value_TEXT:
			v = a.GetText().GetValue()
		case mesos.Value_SCALAR:
			v = strconv.FormatFloat(a.GetScalar().GetValue(), 'G', -1, 64)
		}

		if !validation.IsQualifiedName(k) {
			log.V(3).Infof("ignoring invalid node label name %q", k)
			continue
		}

		if !validation.IsValidLabelValue(v) {
			log.V(3).Infof("ignoring invalid node label %s value: %q", k, v)
			continue
		}

		l[k] = v
	}
	return l
}
