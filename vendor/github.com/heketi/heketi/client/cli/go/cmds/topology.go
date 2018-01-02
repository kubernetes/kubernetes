//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package cmds

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"

	client "github.com/heketi/heketi/client/api/go-client"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/spf13/cobra"
)

const (
	DURABILITY_STRING_REPLICATE       = "replicate"
	DURABILITY_STRING_DISTRIBUTE_ONLY = "none"
	DURABILITY_STRING_EC              = "disperse"
)

var jsonConfigFile string

// Config file
type ConfigFileNode struct {
	Devices []string           `json:"devices"`
	Node    api.NodeAddRequest `json:"node"`
}
type ConfigFileCluster struct {
	Nodes []ConfigFileNode `json:"nodes"`
}
type ConfigFile struct {
	Clusters []ConfigFileCluster `json:"clusters"`
}

func init() {
	RootCmd.AddCommand(topologyCommand)
	topologyCommand.AddCommand(topologyLoadCommand)
	topologyCommand.AddCommand(topologyInfoCommand)
	topologyLoadCommand.Flags().StringVarP(&jsonConfigFile, "json", "j", "",
		"\n\tConfiguration containing devices, nodes, and clusters, in"+
			"\n\tJSON format.")
	topologyLoadCommand.SilenceUsage = true
	topologyInfoCommand.SilenceUsage = true
}

var topologyCommand = &cobra.Command{
	Use:   "topology",
	Short: "Heketi Topology Management",
	Long:  "Heketi Topology management",
}

func getNodeIdFromHeketiTopology(t *api.TopologyInfoResponse,
	managmentHostName string) *api.NodeInfoResponse {

	for _, c := range t.ClusterList {
		for _, n := range c.Nodes {
			if n.Hostnames.Manage[0] == managmentHostName {
				return &n
			}
		}
	}

	return nil
}

func getDeviceIdFromHeketiTopology(t *api.TopologyInfoResponse,
	managmentHostName string,
	deviceName string) *api.DeviceInfoResponse {

	for _, c := range t.ClusterList {
		for _, n := range c.Nodes {
			if n.Hostnames.Manage[0] == managmentHostName {
				for _, d := range n.DevicesInfo {
					if d.Name == deviceName {
						return &d
					}
				}
			}
		}
	}

	return nil
}

var topologyLoadCommand = &cobra.Command{
	Use:     "load",
	Short:   "Add devices to Heketi from a configuration file",
	Long:    "Add devices to Heketi from a configuration file",
	Example: " $ heketi-cli topology load --json=topo.json",
	RunE: func(cmd *cobra.Command, args []string) error {

		// Check arguments
		if jsonConfigFile == "" {
			return errors.New("Missing configuration file")
		}

		// Load config file
		fp, err := os.Open(jsonConfigFile)
		if err != nil {
			return errors.New("Unable to open config file")
		}
		defer fp.Close()
		configParser := json.NewDecoder(fp)
		var topology ConfigFile
		if err = configParser.Decode(&topology); err != nil {
			return errors.New("Unable to parse config file")
		}

		// Create client
		heketi := client.NewClient(options.Url, options.User, options.Key)

		// Load current topolgy
		heketiTopology, err := heketi.TopologyInfo()
		if err != nil {
			return fmt.Errorf("Unable to get topology information: %v", err)
		}

		// Register topology
		for _, cluster := range topology.Clusters {

			// Register Nodes
			var clusterInfo *api.ClusterInfoResponse
			for _, node := range cluster.Nodes {
				// Check node already exists
				nodeInfo := getNodeIdFromHeketiTopology(heketiTopology, node.Node.Hostnames.Manage[0])

				if nodeInfo != nil {
					var err error
					fmt.Fprintf(stdout, "\tFound node %v on cluster %v\n",
						node.Node.Hostnames.Manage[0], nodeInfo.ClusterId)
					clusterInfo, err = heketi.ClusterInfo(nodeInfo.ClusterId)
					if err != nil {
						fmt.Fprintf(stdout, "Unable to get cluster information\n")
						return fmt.Errorf("Unable to get cluster information")
					}
				} else {
					var err error

					// See if we need to create a cluster
					if clusterInfo == nil {
						fmt.Fprintf(stdout, "Creating cluster ... ")
						clusterInfo, err = heketi.ClusterCreate()
						if err != nil {
							return err
						}
						fmt.Fprintf(stdout, "ID: %v\n", clusterInfo.Id)

						// Create a cleanup function in case no
						// nodes or devices are created
						defer func() {
							// Get cluster information
							info, err := heketi.ClusterInfo(clusterInfo.Id)

							// Delete empty cluster
							if err == nil && len(info.Nodes) == 0 && len(info.Volumes) == 0 {
								heketi.ClusterDelete(clusterInfo.Id)
							}
						}()
					}

					// Create node
					fmt.Fprintf(stdout, "\tCreating node %v ... ", node.Node.Hostnames.Manage[0])
					node.Node.ClusterId = clusterInfo.Id
					nodeInfo, err = heketi.NodeAdd(&node.Node)
					if err != nil {
						fmt.Fprintf(stdout, "Unable to create node: %v\n", err)

						// Go to next node
						continue
					} else {
						fmt.Fprintf(stdout, "ID: %v\n", nodeInfo.Id)
					}
				}

				// Add devices
				for _, device := range node.Devices {
					deviceInfo := getDeviceIdFromHeketiTopology(heketiTopology,
						nodeInfo.Hostnames.Manage[0],
						device)
					if deviceInfo != nil {
						fmt.Fprintf(stdout, "\t\tFound device %v\n", device)
					} else {
						fmt.Fprintf(stdout, "\t\tAdding device %v ... ", device)

						req := &api.DeviceAddRequest{}
						req.Name = device
						req.NodeId = nodeInfo.Id
						err := heketi.DeviceAdd(req)
						if err != nil {
							fmt.Fprintf(stdout, "Unable to add device: %v\n", err)
						} else {
							fmt.Fprintf(stdout, "OK\n")
						}
					}
				}
			}
		}
		return nil
	},
}

var topologyInfoCommand = &cobra.Command{
	Use:     "info",
	Short:   "Retreives information about the current Topology",
	Long:    "Retreives information about the current Topology",
	Example: " $ heketi-cli topology info",
	RunE: func(cmd *cobra.Command, args []string) error {

		// Create a client to talk to Heketi
		heketi := client.NewClient(options.Url, options.User, options.Key)

		// Create Topology
		topoinfo, err := heketi.TopologyInfo()
		if err != nil {
			return err
		}

		// Check if JSON should be printed
		if options.Json {
			data, err := json.Marshal(topoinfo)
			if err != nil {
				return err
			}
			fmt.Fprintf(stdout, string(data))
		} else {

			// Get the cluster list and iterate over
			for i, _ := range topoinfo.ClusterList {
				fmt.Fprintf(stdout, "\nCluster Id: %v\n", topoinfo.ClusterList[i].Id)
				fmt.Fprintf(stdout, "\n    %s\n", "Volumes:")
				for k, _ := range topoinfo.ClusterList[i].Volumes {

					// Format and print volumeinfo  on this cluster
					v := topoinfo.ClusterList[i].Volumes[k]
					s := fmt.Sprintf("\n\tName: %v\n"+
						"\tSize: %v\n"+
						"\tId: %v\n"+
						"\tCluster Id: %v\n"+
						"\tMount: %v\n"+
						"\tMount Options: backup-volfile-servers=%v\n"+
						"\tDurability Type: %v\n",
						v.Name,
						v.Size,
						v.Id,
						v.Cluster,
						v.Mount.GlusterFS.MountPoint,
						v.Mount.GlusterFS.Options["backup-volfile-servers"],
						v.Durability.Type)

					switch v.Durability.Type {
					case api.DurabilityEC:
						s += fmt.Sprintf("\tDisperse Data: %v\n"+
							"\tDisperse Redundancy: %v\n",
							v.Durability.Disperse.Data,
							v.Durability.Disperse.Redundancy)
					case api.DurabilityReplicate:
						s += fmt.Sprintf("\tReplica: %v\n",
							v.Durability.Replicate.Replica)
					}
					if v.Snapshot.Enable {
						s += fmt.Sprintf("\tSnapshot: Enabled\n"+
							"\tSnapshot Factor: %.2f\n",
							v.Snapshot.Factor)
					} else {
						s += "\tSnapshot: Disabled\n"
					}
					s += "\n\t\tBricks:\n"
					for _, b := range v.Bricks {
						s += fmt.Sprintf("\t\t\tId: %v\n"+
							"\t\t\tPath: %v\n"+
							"\t\t\tSize (GiB): %v\n"+
							"\t\t\tNode: %v\n"+
							"\t\t\tDevice: %v\n\n",
							b.Id,
							b.Path,
							b.Size/(1024*1024),
							b.NodeId,
							b.DeviceId)
					}
					fmt.Fprintf(stdout, "%s", s)
				}

				// format and print each Node information on this cluster
				fmt.Fprintf(stdout, "\n    %s\n", "Nodes:")
				for j, _ := range topoinfo.ClusterList[i].Nodes {
					info := topoinfo.ClusterList[i].Nodes[j]
					fmt.Fprintf(stdout, "\n\tNode Id: %v\n"+
						"\tState: %v\n"+
						"\tCluster Id: %v\n"+
						"\tZone: %v\n"+
						"\tManagement Hostname: %v\n"+
						"\tStorage Hostname: %v\n",
						info.Id,
						info.State,
						info.ClusterId,
						info.Zone,
						info.Hostnames.Manage[0],
						info.Hostnames.Storage[0])
					fmt.Fprintf(stdout, "\tDevices:\n")

					// format and print the device info
					for j, d := range info.DevicesInfo {
						fmt.Fprintf(stdout, "\t\tId:%-35v"+
							"Name:%-20v"+
							"State:%-10v"+
							"Size (GiB):%-8v"+
							"Used (GiB):%-8v"+
							"Free (GiB):%-8v\n",
							d.Id,
							d.Name,
							d.State,
							d.Storage.Total/(1024*1024),
							d.Storage.Used/(1024*1024),
							d.Storage.Free/(1024*1024))

						// format and print the brick information
						fmt.Fprintf(stdout, "\t\t\tBricks:\n")
						for _, d := range info.DevicesInfo[j].Bricks {
							fmt.Fprintf(stdout, "\t\t\t\tId:%-35v"+
								"Size (GiB):%-8v"+
								"Path: %v\n",
								d.Id,
								d.Size/(1024*1024),
								d.Path)
						}
					}
				}
			}

		}

		return nil
	},
}
