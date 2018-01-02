package main

import (
	"runtime"

	"github.com/docker/docker/daemon/config"
	"github.com/docker/docker/opts"
	"github.com/spf13/pflag"
)

const (
	// defaultShutdownTimeout is the default shutdown timeout for the daemon
	defaultShutdownTimeout = 15
	// defaultTrustKeyFile is the default filename for the trust key
	defaultTrustKeyFile = "key.json"
)

// installCommonConfigFlags adds flags to the pflag.FlagSet to configure the daemon
func installCommonConfigFlags(conf *config.Config, flags *pflag.FlagSet) {
	var maxConcurrentDownloads, maxConcurrentUploads int

	conf.ServiceOptions.InstallCliFlags(flags)

	flags.Var(opts.NewNamedListOptsRef("storage-opts", &conf.GraphOptions, nil), "storage-opt", "Storage driver options")
	flags.Var(opts.NewNamedListOptsRef("authorization-plugins", &conf.AuthorizationPlugins, nil), "authorization-plugin", "Authorization plugins to load")
	flags.Var(opts.NewNamedListOptsRef("exec-opts", &conf.ExecOptions, nil), "exec-opt", "Runtime execution options")
	flags.StringVarP(&conf.Pidfile, "pidfile", "p", defaultPidFile, "Path to use for daemon PID file")
	flags.StringVarP(&conf.Root, "graph", "g", defaultDataRoot, "Root of the Docker runtime")

	// "--graph" is "soft-deprecated" in favor of "data-root". This flag was added
	// before Docker 1.0, so won't be removed, only hidden, to discourage its usage.
	flags.MarkHidden("graph")

	flags.StringVar(&conf.Root, "data-root", defaultDataRoot, "Root directory of persistent Docker state")

	flags.BoolVarP(&conf.AutoRestart, "restart", "r", true, "--restart on the daemon has been deprecated in favor of --restart policies on docker run")
	flags.MarkDeprecated("restart", "Please use a restart policy on docker run")

	// Windows doesn't support setting the storage driver - there is no choice as to which ones to use.
	if runtime.GOOS != "windows" {
		flags.StringVarP(&conf.GraphDriver, "storage-driver", "s", "", "Storage driver to use")
	}

	flags.IntVar(&conf.Mtu, "mtu", 0, "Set the containers network MTU")
	flags.BoolVar(&conf.RawLogs, "raw-logs", false, "Full timestamps without ANSI coloring")
	flags.Var(opts.NewListOptsRef(&conf.DNS, opts.ValidateIPAddress), "dns", "DNS server to use")
	flags.Var(opts.NewNamedListOptsRef("dns-opts", &conf.DNSOptions, nil), "dns-opt", "DNS options to use")
	flags.Var(opts.NewListOptsRef(&conf.DNSSearch, opts.ValidateDNSSearch), "dns-search", "DNS search domains to use")
	flags.Var(opts.NewNamedListOptsRef("labels", &conf.Labels, opts.ValidateLabel), "label", "Set key=value labels to the daemon")
	flags.StringVar(&conf.LogConfig.Type, "log-driver", "json-file", "Default driver for container logs")
	flags.Var(opts.NewNamedMapOpts("log-opts", conf.LogConfig.Config, nil), "log-opt", "Default log driver options for containers")
	flags.StringVar(&conf.ClusterAdvertise, "cluster-advertise", "", "Address or interface name to advertise")
	flags.StringVar(&conf.ClusterStore, "cluster-store", "", "URL of the distributed storage backend")
	flags.Var(opts.NewNamedMapOpts("cluster-store-opts", conf.ClusterOpts, nil), "cluster-store-opt", "Set cluster store options")
	flags.StringVar(&conf.CorsHeaders, "api-cors-header", "", "Set CORS headers in the Engine API")
	flags.IntVar(&maxConcurrentDownloads, "max-concurrent-downloads", config.DefaultMaxConcurrentDownloads, "Set the max concurrent downloads for each pull")
	flags.IntVar(&maxConcurrentUploads, "max-concurrent-uploads", config.DefaultMaxConcurrentUploads, "Set the max concurrent uploads for each push")
	flags.IntVar(&conf.ShutdownTimeout, "shutdown-timeout", defaultShutdownTimeout, "Set the default shutdown timeout")

	flags.StringVar(&conf.SwarmDefaultAdvertiseAddr, "swarm-default-advertise-addr", "", "Set default address or interface for swarm advertised address")
	flags.BoolVar(&conf.Experimental, "experimental", false, "Enable experimental features")

	flags.StringVar(&conf.MetricsAddress, "metrics-addr", "", "Set default address and port to serve the metrics api on")

	flags.StringVar(&conf.NodeGenericResources, "node-generic-resources", "", "user defined resources (e.g. fpga=2;gpu={UUID1,UUID2,UUID3})")
	flags.IntVar(&conf.NetworkControlPlaneMTU, "network-control-plane-mtu", config.DefaultNetworkMtu, "Network Control plane MTU")

	// "--deprecated-key-path" is to allow configuration of the key used
	// for the daemon ID and the deprecated image signing. It was never
	// exposed as a command line option but is added here to allow
	// overriding the default path in configuration.
	flags.Var(opts.NewQuotedString(&conf.TrustKeyPath), "deprecated-key-path", "Path to key file for ID and image signing")
	flags.MarkHidden("deprecated-key-path")

	conf.MaxConcurrentDownloads = &maxConcurrentDownloads
	conf.MaxConcurrentUploads = &maxConcurrentUploads
}
