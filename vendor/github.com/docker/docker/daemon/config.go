package daemon

import (
	"github.com/docker/docker/opts"
	flag "github.com/docker/docker/pkg/mflag"
	"github.com/docker/docker/runconfig"
)

const (
	defaultNetworkMtu    = 1500
	disableNetworkBridge = "none"
)

// CommonConfig defines the configuration of a docker daemon which are
// common across platforms.
type CommonConfig struct {
	AutoRestart    bool
	Bridge         bridgeConfig // Bridge holds bridge network specific configuration.
	Context        map[string][]string
	CorsHeaders    string
	DisableBridge  bool
	Dns            []string
	DnsSearch      []string
	EnableCors     bool
	ExecDriver     string
	ExecOptions    []string
	ExecRoot       string
	GraphDriver    string
	GraphOptions   []string
	Labels         []string
	LogConfig      runconfig.LogConfig
	Mtu            int
	Pidfile        string
	Root           string
	TrustKeyPath   string
	DefaultNetwork string
	NetworkKVStore string
}

// InstallCommonFlags adds command-line options to the top-level flag parser for
// the current process.
// Subsequent calls to `flag.Parse` will populate config with values parsed
// from the command-line.
func (config *Config) InstallCommonFlags() {
	opts.ListVar(&config.GraphOptions, []string{"-storage-opt"}, "Set storage driver options")
	opts.ListVar(&config.ExecOptions, []string{"-exec-opt"}, "Set exec driver options")
	flag.StringVar(&config.Pidfile, []string{"p", "-pidfile"}, defaultPidFile, "Path to use for daemon PID file")
	flag.StringVar(&config.Root, []string{"g", "-graph"}, defaultGraph, "Root of the Docker runtime")
	flag.StringVar(&config.ExecRoot, []string{"-exec-root"}, "/var/run/docker", "Root of the Docker execdriver")
	flag.BoolVar(&config.AutoRestart, []string{"#r", "#-restart"}, true, "--restart on the daemon has been deprecated in favor of --restart policies on docker run")
	flag.StringVar(&config.GraphDriver, []string{"s", "-storage-driver"}, "", "Storage driver to use")
	flag.StringVar(&config.ExecDriver, []string{"e", "-exec-driver"}, defaultExec, "Exec driver to use")
	flag.IntVar(&config.Mtu, []string{"#mtu", "-mtu"}, 0, "Set the containers network MTU")
	flag.BoolVar(&config.EnableCors, []string{"#api-enable-cors", "#-api-enable-cors"}, false, "Enable CORS headers in the remote API, this is deprecated by --api-cors-header")
	flag.StringVar(&config.CorsHeaders, []string{"-api-cors-header"}, "", "Set CORS headers in the remote API")
	// FIXME: why the inconsistency between "hosts" and "sockets"?
	opts.IPListVar(&config.Dns, []string{"#dns", "-dns"}, "DNS server to use")
	opts.DNSSearchListVar(&config.DnsSearch, []string{"-dns-search"}, "DNS search domains to use")
	opts.LabelListVar(&config.Labels, []string{"-label"}, "Set key=value labels to the daemon")
	flag.StringVar(&config.LogConfig.Type, []string{"-log-driver"}, "json-file", "Default driver for container logs")
	opts.LogOptsVar(config.LogConfig.Config, []string{"-log-opt"}, "Set log driver options")
}
