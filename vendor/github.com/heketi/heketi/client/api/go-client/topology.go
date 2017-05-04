//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), as published by the Free Software Foundation,
// or under the Apache License, Version 2.0 <LICENSE-APACHE2 or
// http://www.apache.org/licenses/LICENSE-2.0>.
//
// You may not use this file except in compliance with those terms.
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
