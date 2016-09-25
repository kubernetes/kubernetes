//
// Copyright (c) 2015 The heketi Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package client

import (
	"github.com/heketi/heketi/pkg/glusterfs/api"
)

func (c *Client) TopologyInfo() (*api.TopologyInfoResponse, error) {
	topo := &api.TopologyInfoResponse{
		ClusterList: make([]api.Cluster, 0),
	}
	clusterlist, err := c.ClusterList()
	if err != nil {
		return nil, err
	}
	for _, cluster := range clusterlist.Clusters {
		clusteri, err := c.ClusterInfo(cluster)
		if err != nil {
			return nil, err
		}
		cluster := api.Cluster{
			Id:      clusteri.Id,
			Volumes: make([]api.VolumeInfoResponse, 0),
			Nodes:   make([]api.NodeInfoResponse, 0),
		}
		cluster.Id = clusteri.Id

		// Iterate over the volume list in the cluster
		for _, volumes := range clusteri.Volumes {
			volumesi, err := c.VolumeInfo(volumes)
			if err != nil {
				return nil, err
			}
			if volumesi.Cluster == cluster.Id {
				cluster.Volumes = append(cluster.Volumes, *volumesi)
			}
		}

		// Iterate over the nodes in the cluster
		for _, node := range clusteri.Nodes {
			nodei, err := c.NodeInfo(string(node))
			if err != nil {
				return nil, err
			}
			cluster.Nodes = append(cluster.Nodes, *nodei)
		}
		topo.ClusterList = append(topo.ClusterList, cluster)
	}
	return topo, nil

}
