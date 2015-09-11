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

package lb

import (
	"strings"

	compute "google.golang.org/api/compute/v1"
	"k8s.io/kubernetes/pkg/util"

	"github.com/golang/glog"
)

type Instances struct {
	*ClusterManager
	// Currently unused
	pool *poolStore
}

func NewInstancePool(c *ClusterManager) *Instances {
	return &Instances{c, newPoolStore()}
}

func (i *Instances) list() (util.StringSet, error) {
	nodeNames := util.NewStringSet()
	instances, err := i.cloud.ListInstancesInInstanceGroup(
		i.defaultIg.Name, allInstances)
	if err != nil {
		return nodeNames, err
	}
	for _, ins := range instances.Items {
		// TODO: If round trips weren't so slow one would be inclided
		// to GetInstance using this url and get the name.
		parts := strings.Split(ins.Instance, "/")
		nodeNames.Insert(parts[len(parts)-1])
	}
	return nodeNames, nil
}

func (i *Instances) create(name string) (*compute.InstanceGroup, error) {
	ig, err := i.cloud.GetInstanceGroup(name)
	if ig != nil {
		glog.Infof("Instance group %v already exists", ig.Name)
		return ig, nil
	}

	glog.Infof("Creating instance group %v", name)
	ig, err = i.cloud.CreateInstanceGroup(name)
	if err != nil {
		return nil, err
	}
	return ig, err
}

func (i *Instances) Get(name string) (*compute.InstanceGroup, error) {
	ig, err := i.cloud.GetInstanceGroup(name)
	if err != nil {
		return nil, err
	}
	return ig, nil
}

func (i *Instances) Add(names []string) error {
	glog.Infof("Adding nodes %v to %v", names, i.defaultIg.Name)
	return i.cloud.AddInstancesToInstanceGroup(i.defaultIg.Name, names)
}

func (i *Instances) Remove(names []string) error {
	glog.Infof("Removing nodes %v", names)
	return i.cloud.RemoveInstancesFromInstanceGroup(i.defaultIg.Name, names)
}

func (i *Instances) Sync(nodes []string) error {
	glog.Infof("Syncing nodes %v", nodes)
	gceNodes, err := i.list()
	if err != nil {
		return err
	}
	kubeNodes := util.NewStringSet(nodes...)
	if err := i.Remove(
		gceNodes.Difference(kubeNodes).List()); err != nil {
		return err
	}
	// Unlink the other resources, we only have a single instance group,
	// and we can't create instances. So the add does not perform an edge hop.
	if err := i.Add(
		kubeNodes.Difference(gceNodes).List()); err != nil {
		return err
	}
	return nil
}
