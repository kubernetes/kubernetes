/*
Copyright 2014 Google Inc. All rights reserved.

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

package controller

import (
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

type MinionController struct {
	cloud           cloudprovider.Interface
	matchRE         string
	staticResources *api.NodeResources
	minions         []string
	kubeClient      client.Interface
}

// NewMinionController returns a new minion controller to sync instances from cloudprovider.
func NewMinionController(
	cloud cloudprovider.Interface,
	matchRE string,
	minions []string,
	staticResources *api.NodeResources,
	kubeClient client.Interface) *MinionController {
	return &MinionController{
		cloud:           cloud,
		matchRE:         matchRE,
		minions:         minions,
		staticResources: staticResources,
		kubeClient:      kubeClient,
	}
}

// Run starts syncing instances from cloudprovider periodically, or create initial minion list.
func (s *MinionController) Run(period time.Duration) {
	if s.cloud != nil && len(s.matchRE) > 0 {
		go util.Forever(func() {
			if err := s.SyncCloud(); err != nil {
				glog.Errorf("Error syncing cloud: %v", err)
			}
		}, period)
	} else {
		go s.SyncStatic(period)
	}
}

// SyncStatic registers list of machines from command line flag. It returns after successful
// registration of all machines.
func (s *MinionController) SyncStatic(period time.Duration) error {
	registered := util.NewStringSet()
	for {
		for _, minionID := range s.minions {
			if registered.Has(minionID) {
				continue
			}
			_, err := s.kubeClient.Nodes().Create(&api.Node{
				ObjectMeta: api.ObjectMeta{Name: minionID},
				Spec: api.NodeSpec{
					Capacity: s.staticResources.Capacity,
				},
			})
			if err == nil {
				registered.Insert(minionID)
			}
		}
		if registered.Len() == len(s.minions) {
			return nil
		}
		time.Sleep(period)
	}
}

// SyncCloud syncs list of instances from cloudprovider to master etcd registry.
func (s *MinionController) SyncCloud() error {
	matches, err := s.cloudMinions()
	if err != nil {
		return err
	}
	minions, err := s.kubeClient.Nodes().List()
	if err != nil {
		return err
	}
	minionMap := make(map[string]*api.Node)
	for _, minion := range minions.Items {
		minionMap[minion.Name] = &minion
	}

	// Create or delete minions from registry.
	for _, minion := range matches.Items {
		if _, ok := minionMap[minion.Name]; !ok {
			glog.Infof("Create minion in registry: %s", minion.Name)
			_, err = s.kubeClient.Nodes().Create(&minion)
			if err != nil {
				glog.Errorf("Create minion error: %s", minion.Name)
			}
		}
		delete(minionMap, minion.Name)
	}

	for minionID := range minionMap {
		glog.Infof("Delete minion from registry: %s", minionID)
		err = s.kubeClient.Nodes().Delete(minionID)
		if err != nil {
			glog.Errorf("Delete minion error: %s", minionID)
		}
	}
	return nil
}

// cloudMinions constructs and returns api.NodeList from cloudprovider.
func (s *MinionController) cloudMinions() (*api.NodeList, error) {
	instances, ok := s.cloud.Instances()
	if !ok {
		return nil, fmt.Errorf("cloud doesn't support instances")
	}
	matches, err := instances.List(s.matchRE)
	if err != nil {
		return nil, err
	}
	result := &api.NodeList{
		Items: make([]api.Node, len(matches)),
	}
	for i := range matches {
		result.Items[i].Name = matches[i]
		hostIP, err := instances.IPAddress(matches[i])
		if err != nil {
			glog.Errorf("error getting instance ip address for %s: %v", matches[i], err)
		} else {
			result.Items[i].Status.HostIP = hostIP.String()
		}
		resources, err := instances.GetNodeResources(matches[i])
		if err != nil {
			return nil, err
		}
		if resources == nil {
			resources = s.staticResources
		}
		if resources != nil {
			result.Items[i].Spec.Capacity = resources.Capacity
		}
	}
	return result, nil
}
