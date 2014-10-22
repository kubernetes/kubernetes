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
		go util.Forever(func() { s.Sync() }, period)
	} else {
		for _, minionID := range s.minions {
			s.kubeClient.Minions().Create(&api.Minion{
				ObjectMeta:    api.ObjectMeta{Name: minionID},
				NodeResources: *s.staticResources,
			})
		}
	}
}

// Sync syncs list of instances from cloudprovider to master etcd registry.
func (s *MinionController) Sync() error {
	matches, err := s.cloudMinions()
	if err != nil {
		return err
	}
	minions, err := s.kubeClient.Minions().List()
	if err != nil {
		return err
	}
	minionMap := make(map[string]*api.Minion)
	for _, minion := range minions.Items {
		minionMap[minion.Name] = &minion
	}

	// Create or delete minions from registry.
	for _, minion := range matches.Items {
		if _, ok := minionMap[minion.Name]; !ok {
			glog.Infof("Create minion in registry: %s", minion.Name)
			s.kubeClient.Minions().Create(&minion)
		}
		delete(minionMap, minion.Name)
	}

	for minionID := range minionMap {
		glog.Infof("Delete minion from registry: %s", minionID)
		s.kubeClient.Minions().Delete(minionID)
	}
	return nil
}

// cloudMinions constructs and returns api.MinionList from cloudprovider.
func (s *MinionController) cloudMinions() (*api.MinionList, error) {
	instances, ok := s.cloud.Instances()
	if !ok {
		return nil, fmt.Errorf("cloud doesn't support instances")
	}
	matches, err := instances.List(s.matchRE)
	if err != nil {
		return nil, err
	}
	result := &api.MinionList{
		Items: make([]api.Minion, len(matches)),
	}
	for i := range matches {
		result.Items[i].Name = matches[i]
		resources, err := instances.GetNodeResources(matches[i])
		if err != nil {
			return nil, err
		}
		if resources == nil {
			resources = s.staticResources
		}
		if resources != nil {
			result.Items[i].NodeResources = *resources
		}
	}
	return result, nil
}
