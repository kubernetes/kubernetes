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
	"time"

	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/queue"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
)

type Registrator interface {
	// Register checks whether the node is registered with the given labels. If it
	// is not, it is created or updated on the apiserver. If an the node was up-to-date,
	// false is returned.
	Register(hostName string, labels map[string]string) (bool, error)

	// Start the registration loop and return immediately.
	Run(terminate <-chan struct{}) error
}

type registration struct {
	hostName string
	labels   map[string]string
}

func (r *registration) Copy() queue.Copyable {
	return &registration{
		hostName: r.hostName,
		labels:   r.labels, // labels are never changed, no need to clone
	}
}

func (r *registration) GetUID() string {
	return r.hostName
}

func (r *registration) Value() queue.UniqueCopyable {
	return r
}

type LookupFunc func(hostName string) *api.Node

type clientRegistrator struct {
	lookupNode LookupFunc
	client     unversionedcore.NodesGetter
	queue      *queue.HistoricalFIFO
}

func NewRegistrator(client unversionedcore.NodesGetter, lookupNode LookupFunc) *clientRegistrator {
	return &clientRegistrator{
		lookupNode: lookupNode,
		client:     client,
		queue:      queue.NewHistorical(nil),
	}
}

func (r *clientRegistrator) Run(terminate <-chan struct{}) error {
	loop := func() {
	RegistrationLoop:
		for {
			obj := r.queue.Pop(terminate)
			log.V(3).Infof("registration event observed")
			if obj == nil {
				break RegistrationLoop
			}
			select {
			case <-terminate:
				break RegistrationLoop
			default:
			}

			rg := obj.(*registration)
			n, needsUpdate := r.updateNecessary(rg.hostName, rg.labels)
			if !needsUpdate {
				log.V(2).Infof("no update needed, skipping for %s: %v", rg.hostName, rg.labels)
				continue
			}

			if n == nil {
				log.V(2).Infof("creating node %s with labels %v", rg.hostName, rg.labels)
				_, err := CreateOrUpdate(r.client, rg.hostName, rg.labels, nil)
				if err != nil {
					log.Errorf("error creating the node %s: %v", rg.hostName, rg.labels)
				}
			} else {
				log.V(2).Infof("updating node %s with labels %v", rg.hostName, rg.labels)
				_, err := Update(r.client, rg.hostName, rg.labels, nil)
				if err != nil && errors.IsNotFound(err) {
					// last chance when our store was out of date
					_, err = Create(r.client, rg.hostName, rg.labels, nil)
				}
				if err != nil {
					log.Errorf("error updating the node %s: %v", rg.hostName, rg.labels)
				}
			}
		}
	}
	go runtime.Until(loop, time.Second, terminate)

	return nil
}

func (r *clientRegistrator) Register(hostName string, labels map[string]string) (bool, error) {
	_, needsUpdate := r.updateNecessary(hostName, labels)

	if needsUpdate {
		log.V(5).Infof("queuing registration for node %s with labels %v", hostName, labels)
		err := r.queue.Update(&registration{
			hostName: hostName,
			labels:   labels,
		})
		if err != nil {
			return false, fmt.Errorf("cannot register node %s: %v", hostName, err)
		}
		return true, nil
	}

	return false, nil
}

// updateNecessary retrieves the node with the given hostname and checks whether the given
// labels would mean any update to the node. The unmodified node is returned, plus
// true iff an update is necessary.
func (r *clientRegistrator) updateNecessary(hostName string, labels map[string]string) (*api.Node, bool) {
	if r.lookupNode == nil {
		return nil, true
	}
	n := r.lookupNode(hostName)
	return n, n == nil || !IsUpToDate(n, labels)
}
