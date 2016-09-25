/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"time"

	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"

	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/util/validation"
)

const (
	labelPrefix         = "k8s.mesosphere.io/attribute-"
	clientRetryCount    = 5
	clientRetryInterval = time.Second
)

// Create creates a new node api object with the given hostname,
// slave attribute labels and annotations
func Create(
	client unversionedcore.NodesGetter,
	hostName string,
	slaveAttrLabels,
	annotations map[string]string,
) (*api.Node, error) {
	n := api.Node{
		ObjectMeta: api.ObjectMeta{
			Name: hostName,
		},
		Spec: api.NodeSpec{
			ExternalID: hostName,
		},
		Status: api.NodeStatus{
			Phase: api.NodePending,
			// WORKAROUND(sttts): make sure that the Ready condition is the
			// first one. The kube-ui v3 depends on this assumption.
			// TODO(sttts): remove this workaround when kube-ui v4 is used or we
			//              merge this with the statusupdate in the controller manager.
			Conditions: []api.NodeCondition{
				{
					Type:              api.NodeReady,
					Status:            api.ConditionTrue,
					Reason:            slaveReadyReason,
					Message:           slaveReadyMessage,
					LastHeartbeatTime: unversioned.Now(),
				},
			},
		},
	}

	n.Labels = mergeMaps(
		map[string]string{"kubernetes.io/hostname": hostName},
		slaveAttrLabels,
	)

	n.Annotations = annotations

	// try to create
	return client.Nodes().Create(&n)
}

// Update updates an existing node api object
// by looking up the given hostname.
// The updated node merges the given slave attribute labels
// and annotations with the found api object.
func Update(
	client unversionedcore.NodesGetter,
	hostname string,
	slaveAttrLabels,
	annotations map[string]string,
) (n *api.Node, err error) {
	for i := 0; i < clientRetryCount; i++ {
		n, err = client.Nodes().Get(hostname)
		if err != nil {
			return nil, fmt.Errorf("error getting node %q: %v", hostname, err)
		}
		if n == nil {
			return nil, fmt.Errorf("no node instance returned for %q", hostname)
		}

		// update labels derived from Mesos slave attributes, keep all other labels
		n.Labels = mergeMaps(
			filterMap(n.Labels, IsNotSlaveAttributeLabel),
			slaveAttrLabels,
		)
		n.Annotations = mergeMaps(n.Annotations, annotations)

		n, err = client.Nodes().Update(n)
		if err == nil && !errors.IsConflict(err) {
			return n, nil
		}

		log.Infof("retry %d/%d: error updating node %v err %v", i, clientRetryCount, n, err)
		time.Sleep(time.Duration(i) * clientRetryInterval)
	}

	return nil, err
}

// CreateOrUpdate creates a node api object or updates an existing one
func CreateOrUpdate(
	client unversionedcore.NodesGetter,
	hostname string,
	slaveAttrLabels,
	annotations map[string]string,
) (*api.Node, error) {
	n, err := Create(client, hostname, slaveAttrLabels, annotations)
	if err == nil {
		return n, nil
	}

	if !errors.IsAlreadyExists(err) {
		return nil, fmt.Errorf("unable to register %q with the apiserver: %v", hostname, err)
	}

	// fall back to update an old node with new labels
	return Update(client, hostname, slaveAttrLabels, annotations)
}

// IsNotSlaveAttributeLabel returns true iff the given label is not derived from a slave attribute
func IsNotSlaveAttributeLabel(key, value string) bool {
	return !IsSlaveAttributeLabel(key, value)
}

// IsSlaveAttributeLabel returns true iff the given label is derived from a slave attribute
func IsSlaveAttributeLabel(key, value string) bool {
	return strings.HasPrefix(key, labelPrefix)
}

// IsUpToDate returns true iff the node's slave labels match the given attributes labels
func IsUpToDate(n *api.Node, labels map[string]string) bool {
	slaveLabels := map[string]string{}
	for k, v := range n.Labels {
		if IsSlaveAttributeLabel(k, "") {
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

		if errs := validation.IsQualifiedName(k); len(errs) != 0 {
			log.V(3).Infof("ignoring invalid node label %q: %v", k, errs)
			continue
		}

		if errs := validation.IsValidLabelValue(v); len(errs) != 0 {
			log.V(3).Infof("ignoring invalid node %s=%q: %v", k, v, errs)
			continue
		}

		l[k] = v
	}
	return l
}

// filterMap filters the given map and returns a new map
// containing all original elements matching the given key-value predicate.
func filterMap(m map[string]string, predicate func(string, string) bool) map[string]string {
	result := make(map[string]string, len(m))
	for k, v := range m {
		if predicate(k, v) {
			result[k] = v
		}
	}
	return result
}

// mergeMaps merges all given maps into a single map.
// There is no advanced key conflict resolution.
// The last key from the given maps wins.
func mergeMaps(ms ...map[string]string) map[string]string {
	var l int
	for _, m := range ms {
		l += len(m)
	}

	result := make(map[string]string, l)
	for _, m := range ms {
		for k, v := range m {
			result[k] = v
		}
	}
	return result
}
