package cli

import (
	"fmt"
	"os"
	"text/tabwriter"

	humanize "github.com/dustin/go-humanize"

	"github.com/codegangsta/cli"

	"github.com/libopenstorage/gossip/types"
	"github.com/libopenstorage/openstorage/api"
	clusterclient "github.com/libopenstorage/openstorage/api/client/cluster"
	"github.com/libopenstorage/openstorage/cluster"
)

type clusterClient struct {
	manager cluster.Cluster
}

func (c *clusterClient) clusterOptions(context *cli.Context) {
	// Currently we choose the default version
	clnt, err := clusterclient.NewClusterClient("", cluster.APIVersion)
	if err != nil {
		fmt.Printf("Failed to initialize client library: %v\n", err)
		os.Exit(1)
	}
	c.manager = clusterclient.ClusterManager(clnt)
}

func (c *clusterClient) status(context *cli.Context) {
	c.clusterOptions(context)
	jsonOut := context.GlobalBool("json")
	outFd := os.Stdout
	fn := "status"

	cluster, err := c.manager.Enumerate()
	if err != nil {
		cmdError(context, fn, err)
		return
	}

	if jsonOut {
		fmtOutput(context, &Format{Cluster: &cluster})
	} else {
		fmt.Fprintf(outFd, "Cluster Information:\nCluster ID %s: Status: %v\n\n",
			cluster.Id, cluster.Status)

		fmt.Fprintf(outFd, "Load Information:\n")
		w := new(tabwriter.Writer)
		w.Init(outFd, 12, 12, 1, ' ', 0)

		fmt.Fprintln(w, "ID\t MGMT IP\t STATUS\t CPU\t MEM TOTAL\t MEM FREE")
		for _, n := range cluster.Nodes {
			status := ""
			if n.Status == api.Status_STATUS_INIT {
				status = "Initializing"
			} else if n.Status == api.Status_STATUS_OK {
				status = "OK"
			} else if n.Status == api.Status_STATUS_OFFLINE {
				status = "Off Line"
			} else {
				status = "Error"
			}

			fmt.Fprintln(w, n.Id, "\t", n.MgmtIp, "\t", status, "\t",
				n.Cpu, "\t", humanize.Bytes(n.MemTotal), "\t",
				humanize.Bytes(n.MemFree))
		}

		fmt.Fprintln(w)
		w.Flush()
	}
}

func (c *clusterClient) inspect(context *cli.Context) {
	c.clusterOptions(context)
	jsonOut := context.GlobalBool("json")
	fn := "inspect"

	cluster, err := c.manager.Enumerate()
	if err != nil {
		cmdError(context, fn, err)
		return
	}

	if jsonOut {
		fmtOutput(context, &Format{Cluster: &cluster})
	} else {
	}
}

func (c *clusterClient) remove(context *cli.Context) {
}

func (c *clusterClient) shutdown(context *cli.Context) {
}

func (c *clusterClient) disableGossip(context *cli.Context) {
	c.clusterOptions(context)
	c.manager.DisableUpdates()
}

func (c *clusterClient) enableGossip(context *cli.Context) {
	c.clusterOptions(context)
	c.manager.EnableUpdates()
}

func (c *clusterClient) gossipStatus(context *cli.Context) {
	c.clusterOptions(context)
	jsonOut := context.GlobalBool("json")
	outFd := os.Stdout

	s := c.manager.GetGossipState()

	if jsonOut {
		fmtOutput(context, &Format{Result: s})
	} else {
		fmt.Println("Individual Node Status")
		w := new(tabwriter.Writer)
		w.Init(outFd, 12, 12, 1, ' ', 0)

		fmt.Fprintln(w, "ID\t LAST UPDATE TS\t STATUS")
		for _, n := range s.NodeStatus {
			statusStr := "Up"
			switch {
			case n.Status == types.NODE_STATUS_DOWN,
				n.Status == types.NODE_STATUS_NOT_IN_QUORUM:
				statusStr = "Down"
			case n.Status == types.NODE_STATUS_INVALID:
				statusStr = "Invalid"
			case n.Status == types.NODE_STATUS_NEVER_GOSSIPED:
				statusStr = "Node not yet gossiped"
			case n.Status == types.NODE_STATUS_SUSPECT_NOT_IN_QUORUM:
				statusStr = "Node Up but not in Quorum."
			}
			fmt.Fprintln(w, n.Id, "\t", n.LastUpdateTs, "\t", statusStr)
		}

		fmt.Fprintln(w)
		w.Flush()
	}
}

// ClusterCommands exports CLI comamnds for File VolumeDriver
func ClusterCommands() []cli.Command {
	c := &clusterClient{}

	commands := []cli.Command{
		{
			Name:    "status",
			Aliases: []string{"s"},
			Usage:   "Inspect the cluster",
			Action:  c.status,
			Flags: []cli.Flag{
				cli.StringFlag{
					Name:  "machine,m",
					Usage: "Comma separated machine ids, e.g uuid1,uuid2",
					Value: "",
				},
			},
		},
		{
			Name:    "inspect",
			Aliases: []string{"l"},
			Usage:   "Inspect nodes in the cluster",
			Action:  c.inspect,
			Flags: []cli.Flag{
				cli.StringFlag{
					Name:  "machine,m",
					Usage: "Comma separated machine ids, e.g uuid1,uuid2",
					Value: "",
				},
			},
		},
		{
			Name:    "disable-gossip",
			Aliases: []string{"dg"},
			Usage:   "Disable gossip updates",
			Action:  c.disableGossip,
		},
		{
			Name:    "enable-gossip",
			Aliases: []string{"eg"},
			Usage:   "Enable gossip updates",
			Action:  c.enableGossip,
		},
		{
			Name:    "gossip-status",
			Aliases: []string{"gs"},
			Usage:   "Display gossip status",
			Action:  c.gossipStatus,
		},
		{
			Name:    "remove",
			Aliases: []string{"r"},
			Usage:   "Remove a machine from the cluster",
			Action:  c.remove,
			Flags: []cli.Flag{
				cli.StringFlag{
					Name:  "machine,m",
					Usage: "Comma separated machine ids, e.g uuid1,uuid2",
					Value: "",
				},
			},
		},
		{
			Name:   "shutdown",
			Usage:  "Shutdown a cluster or a specific machine",
			Action: c.shutdown,
			Flags: []cli.Flag{
				cli.StringFlag{
					Name:  "machine,m",
					Usage: "Comma separated machine ids, e.g uuid1,uuid2",
					Value: "",
				},
			},
		},
	}
	return commands
}
