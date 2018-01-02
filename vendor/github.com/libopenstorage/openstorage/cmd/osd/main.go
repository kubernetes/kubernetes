package main

import (
	"fmt"
	"net/url"
	"os"
	"runtime"
	"strconv"

	"go.pedge.io/dlog"

	"github.com/codegangsta/cli"
	"github.com/docker/docker/pkg/reexec"
	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/api/server"
	"github.com/libopenstorage/openstorage/api/flexvolume"
	osdcli "github.com/libopenstorage/openstorage/cli"
	"github.com/libopenstorage/openstorage/cluster"
	"github.com/libopenstorage/openstorage/config"
	"github.com/libopenstorage/openstorage/graph/drivers"
	"github.com/libopenstorage/openstorage/volume"
	"github.com/libopenstorage/openstorage/volume/drivers"
	"github.com/portworx/kvdb"
	"github.com/portworx/kvdb/consul"
	etcd "github.com/portworx/kvdb/etcd/v2"
	"github.com/portworx/kvdb/mem"
)

var (
	datastores = []string{mem.Name, etcd.Name, consul.Name}
)

func main() {
	if reexec.Init() {
		return
	}
	app := cli.NewApp()
	app.Name = "osd"
	app.Usage = "Open Storage CLI"
	app.Version = config.Version
	app.Flags = []cli.Flag{
		cli.BoolFlag{
			Name:  "json,j",
			Usage: "output in json",
		},
		cli.BoolFlag{
			Name:  osdcli.DaemonAlias,
			Usage: "Start OSD in daemon mode",
		},
		cli.StringSliceFlag{
			Name:  "driver",
			Usage: "driver name and options: name=btrfs,home=/var/openstorage/btrfs",
			Value: new(cli.StringSlice),
		},
		cli.StringFlag{
			Name:  "kvdb,k",
			Usage: "uri to kvdb e.g. kv-mem://localhost, etcd-kv://localhost:4001, consul-kv://localhost:8500",
			Value: "kv-mem://localhost",
		},
		cli.StringFlag{
			Name:  "file,f",
			Usage: "file to read the OSD configuration from.",
			Value: "",
		},
	}
	app.Action = wrapAction(start)
	app.Commands = []cli.Command{
		{
			Name:        "driver",
			Aliases:     []string{"d"},
			Usage:       "Manage drivers",
			Subcommands: osdcli.DriverCommands(),
		},
		{
			Name:        "cluster",
			Aliases:     []string{"c"},
			Usage:       "Manage cluster",
			Subcommands: osdcli.ClusterCommands(),
		},
		{
			Name:    "version",
			Aliases: []string{"v"},
			Usage:   "Display version",
			Action:  wrapAction(showVersion),
		},
	}

	// Register all volume drivers with the CLI.
	for _, v := range volumedrivers.AllDrivers {
		// TODO(pedge): was an and, but we have no drivers that have two types
		switch v.DriverType {
		case api.DriverType_DRIVER_TYPE_BLOCK:
			bCmds := osdcli.BlockVolumeCommands(v.Name)
			cmds := append(bCmds)
			c := cli.Command{
				Name:        v.Name,
				Usage:       fmt.Sprintf("Manage %s storage", v.Name),
				Subcommands: cmds,
			}
			app.Commands = append(app.Commands, c)
		case api.DriverType_DRIVER_TYPE_FILE:
			fCmds := osdcli.FileVolumeCommands(v.Name)
			cmds := append(fCmds)
			c := cli.Command{
				Name:        v.Name,
				Usage:       fmt.Sprintf("Manage %s volumes", v.Name),
				Subcommands: cmds,
			}
			app.Commands = append(app.Commands, c)
		}
	}

	// Register all graph drivers with the CLI.
	for _, v := range graphdrivers.AllDrivers {
		// TODO(pedge): was an and, but we have no drivers that have two types
		switch v.DriverType {
		case api.DriverType_DRIVER_TYPE_GRAPH:
			cmds := osdcli.GraphDriverCommands(v.Name)
			c := cli.Command{
				Name:        v.Name,
				Usage:       fmt.Sprintf("Manage %s graph storage", v.Name),
				Subcommands: cmds,
			}
			app.Commands = append(app.Commands, c)
		}
	}

	app.Run(os.Args)
}

func start(c *cli.Context) error {
	if !osdcli.DaemonMode(c) {
		cli.ShowAppHelp(c)
		return nil
	}

	// We are in daemon mode.
	file := c.String("file")
	if file == "" {
		return fmt.Errorf("OSD configuration file not specified.  Visit openstorage.org for an example.")
	}

	cfg, err := config.Parse(file)
	if err != nil {
		return err
	}
	kvdbURL := c.String("kvdb")
	u, err := url.Parse(kvdbURL)
	scheme := u.Scheme
	u.Scheme = "http"

	kv, err := kvdb.New(scheme, "openstorage", []string{u.String()}, nil, dlog.Panicf)
	if err != nil {
		return fmt.Errorf("Failed to initialize KVDB: %v (%v)\nSupported datastores: %v", scheme, err, datastores)
	}
	if err := kvdb.SetInstance(kv); err != nil {
		return fmt.Errorf("Failed to initialize KVDB: %v", err)
	}

	// Start the cluster state machine, if enabled.
	clusterInit := false
	if cfg.Osd.ClusterConfig.NodeId != "" && cfg.Osd.ClusterConfig.ClusterId != "" {
		dlog.Infof("OSD enabling cluster mode.")
		if err := cluster.Init(cfg.Osd.ClusterConfig); err != nil {
			return fmt.Errorf("Unable to init cluster server: %v", err)
		}
		if err := server.StartClusterAPI(cluster.APIBase, 0); err != nil {
			return fmt.Errorf("Unable to start cluster API server: %v", err)
		}
		clusterInit = true
	}

	isDefaultSet := false
	// Start the volume drivers.
	for d, v := range cfg.Osd.Drivers {
		dlog.Infof("Starting volume driver: %v", d)
		if err := volumedrivers.Register(d, v); err != nil {
			return fmt.Errorf("Unable to start volume driver: %v, %v", d, err)
		}

		var mgmtPort, pluginPort uint64
		if port, ok := v[config.MgmtPortKey]; ok {
			mgmtPort, err = strconv.ParseUint(port, 10, 16)
			if err != nil {
				return fmt.Errorf("Invalid OSD Config File. Invalid Mgmt Port number for Driver : %s", d)
			}
		} else {
			mgmtPort = 0
		}

		if port, ok := v[config.PluginPortKey]; ok {
			pluginPort, err = strconv.ParseUint(port, 10, 16)
			if err != nil {
				return fmt.Errorf("Invalid OSD Config File. Invalid Plugin Port number for Driver : %s", d)
			}
		} else {
			pluginPort = 0
		}

		if err := server.StartPluginAPI(
			d,
			volume.DriverAPIBase,
			volume.PluginAPIBase,
			uint16(mgmtPort),
			uint16(pluginPort),
		); err != nil {
			return fmt.Errorf("Unable to start volume plugin: %v", err)
		}
		if d != "" && cfg.Osd.ClusterConfig.DefaultDriver == d {
			isDefaultSet = true
		}
	}

	if cfg.Osd.ClusterConfig.DefaultDriver != "" && !isDefaultSet {
		return fmt.Errorf("Invalid OSD config file: Default Driver specified but driver not initialized")
	}

	if err := flexvolume.StartFlexVolumeAPI(config.FlexVolumePort, cfg.Osd.ClusterConfig.DefaultDriver); err != nil {
		return fmt.Errorf("Unable to start flexvolume API: %v", err)
	}

	// Start the graph drivers.
	for d := range cfg.Osd.GraphDrivers {
		dlog.Infof("Starting graph driver: %v", d)
		if err := server.StartGraphAPI(d, volume.PluginAPIBase); err != nil {
			return fmt.Errorf("Unable to start graph plugin: %v", err)
		}
	}

	if clusterInit {
		cm, err := cluster.Inst()
		if err != nil {
			return fmt.Errorf("Unable to find cluster instance: %v", err)
		}
		if err := cm.Start(0, false); err != nil {
			return fmt.Errorf("Unable to start cluster manager: %v", err)
		}
	}

	// Daemon does not exit.
	select {}
}

func showVersion(c *cli.Context) error {
	fmt.Println("OSD Version:", config.Version)
	fmt.Println("Go Version:", runtime.Version())
	fmt.Println("OS:", runtime.GOOS)
	fmt.Println("Arch:", runtime.GOARCH)
	return nil
}

func wrapAction(f func(*cli.Context) error) func(*cli.Context) {
	return func(c *cli.Context) {
		if err := f(c); err != nil {
			dlog.Warnln(err.Error())
			os.Exit(1)
		}
	}
}
