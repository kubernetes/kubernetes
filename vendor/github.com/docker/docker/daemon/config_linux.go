package daemon

import (
	"net"

	"github.com/docker/docker/opts"
	flag "github.com/docker/docker/pkg/mflag"
	"github.com/docker/docker/pkg/ulimit"
)

var (
	defaultPidFile = "/var/run/docker.pid"
	defaultGraph   = "/var/lib/docker"
	defaultExec    = "native"
)

// Config defines the configuration of a docker daemon.
// These are the configuration settings that you pass
// to the docker daemon when you launch it with say: `docker -d -e lxc`
type Config struct {
	CommonConfig

	// Fields below here are platform specific.

	EnableSelinuxSupport bool
	SocketGroup          string
	Ulimits              map[string]*ulimit.Ulimit
}

// bridgeConfig stores all the bridge driver specific
// configuration.
type bridgeConfig struct {
	EnableIPv6                  bool
	EnableIPTables              bool
	EnableIPForward             bool
	EnableIPMasq                bool
	EnableUserlandProxy         bool
	DefaultIP                   net.IP
	Iface                       string
	IP                          string
	FixedCIDR                   string
	FixedCIDRv6                 string
	DefaultGatewayIPv4          net.IP
	DefaultGatewayIPv6          net.IP
	InterContainerCommunication bool
}

// InstallFlags adds command-line options to the top-level flag parser for
// the current process.
// Subsequent calls to `flag.Parse` will populate config with values parsed
// from the command-line.
func (config *Config) InstallFlags() {
	// First handle install flags which are consistent cross-platform
	config.InstallCommonFlags()

	// Then platform-specific install flags
	flag.BoolVar(&config.EnableSelinuxSupport, []string{"-selinux-enabled"}, false, "Enable selinux support")
	flag.StringVar(&config.SocketGroup, []string{"G", "-group"}, "docker", "Group for the unix socket")
	config.Ulimits = make(map[string]*ulimit.Ulimit)
	opts.UlimitMapVar(config.Ulimits, []string{"-default-ulimit"}, "Set default ulimits for containers")
	flag.BoolVar(&config.Bridge.EnableIPTables, []string{"#iptables", "-iptables"}, true, "Enable addition of iptables rules")
	flag.BoolVar(&config.Bridge.EnableIPForward, []string{"#ip-forward", "-ip-forward"}, true, "Enable net.ipv4.ip_forward")
	flag.BoolVar(&config.Bridge.EnableIPMasq, []string{"-ip-masq"}, true, "Enable IP masquerading")
	flag.BoolVar(&config.Bridge.EnableIPv6, []string{"-ipv6"}, false, "Enable IPv6 networking")
	flag.StringVar(&config.Bridge.IP, []string{"#bip", "-bip"}, "", "Specify network bridge IP")
	flag.StringVar(&config.Bridge.Iface, []string{"b", "-bridge"}, "", "Attach containers to a network bridge")
	flag.StringVar(&config.Bridge.FixedCIDR, []string{"-fixed-cidr"}, "", "IPv4 subnet for fixed IPs")
	flag.StringVar(&config.Bridge.FixedCIDRv6, []string{"-fixed-cidr-v6"}, "", "IPv6 subnet for fixed IPs")
	opts.IPVar(&config.Bridge.DefaultGatewayIPv4, []string{"-default-gateway"}, "", "Container default gateway IPv4 address")
	opts.IPVar(&config.Bridge.DefaultGatewayIPv6, []string{"-default-gateway-v6"}, "", "Container default gateway IPv6 address")
	flag.BoolVar(&config.Bridge.InterContainerCommunication, []string{"#icc", "-icc"}, true, "Enable inter-container communication")
	opts.IPVar(&config.Bridge.DefaultIP, []string{"#ip", "-ip"}, "0.0.0.0", "Default IP when binding container ports")
	flag.BoolVar(&config.Bridge.EnableUserlandProxy, []string{"-userland-proxy"}, true, "Use userland proxy for loopback traffic")
	config.attachExperimentalFlags()
}
