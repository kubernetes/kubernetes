package config

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"reflect"
	"runtime"
	"strings"
	"sync"

	daemondiscovery "github.com/docker/docker/daemon/discovery"
	"github.com/docker/docker/opts"
	"github.com/docker/docker/pkg/authorization"
	"github.com/docker/docker/pkg/discovery"
	"github.com/docker/docker/registry"
	"github.com/imdario/mergo"
	"github.com/sirupsen/logrus"
	"github.com/spf13/pflag"
)

const (
	// DefaultMaxConcurrentDownloads is the default value for
	// maximum number of downloads that
	// may take place at a time for each pull.
	DefaultMaxConcurrentDownloads = 3
	// DefaultMaxConcurrentUploads is the default value for
	// maximum number of uploads that
	// may take place at a time for each push.
	DefaultMaxConcurrentUploads = 5
	// StockRuntimeName is the reserved name/alias used to represent the
	// OCI runtime being shipped with the docker daemon package.
	StockRuntimeName = "runc"
	// DefaultShmSize is the default value for container's shm size
	DefaultShmSize = int64(67108864)
	// DefaultNetworkMtu is the default value for network MTU
	DefaultNetworkMtu = 1500
	// DisableNetworkBridge is the default value of the option to disable network bridge
	DisableNetworkBridge = "none"
	// DefaultInitBinary is the name of the default init binary
	DefaultInitBinary = "docker-init"
)

// flatOptions contains configuration keys
// that MUST NOT be parsed as deep structures.
// Use this to differentiate these options
// with others like the ones in CommonTLSOptions.
var flatOptions = map[string]bool{
	"cluster-store-opts": true,
	"log-opts":           true,
	"runtimes":           true,
	"default-ulimits":    true,
}

// LogConfig represents the default log configuration.
// It includes json tags to deserialize configuration from a file
// using the same names that the flags in the command line use.
type LogConfig struct {
	Type   string            `json:"log-driver,omitempty"`
	Config map[string]string `json:"log-opts,omitempty"`
}

// commonBridgeConfig stores all the platform-common bridge driver specific
// configuration.
type commonBridgeConfig struct {
	Iface     string `json:"bridge,omitempty"`
	FixedCIDR string `json:"fixed-cidr,omitempty"`
}

// CommonTLSOptions defines TLS configuration for the daemon server.
// It includes json tags to deserialize configuration from a file
// using the same names that the flags in the command line use.
type CommonTLSOptions struct {
	CAFile   string `json:"tlscacert,omitempty"`
	CertFile string `json:"tlscert,omitempty"`
	KeyFile  string `json:"tlskey,omitempty"`
}

// CommonConfig defines the configuration of a docker daemon which is
// common across platforms.
// It includes json tags to deserialize configuration from a file
// using the same names that the flags in the command line use.
type CommonConfig struct {
	AuthzMiddleware       *authorization.Middleware `json:"-"`
	AuthorizationPlugins  []string                  `json:"authorization-plugins,omitempty"` // AuthorizationPlugins holds list of authorization plugins
	AutoRestart           bool                      `json:"-"`
	Context               map[string][]string       `json:"-"`
	DisableBridge         bool                      `json:"-"`
	DNS                   []string                  `json:"dns,omitempty"`
	DNSOptions            []string                  `json:"dns-opts,omitempty"`
	DNSSearch             []string                  `json:"dns-search,omitempty"`
	ExecOptions           []string                  `json:"exec-opts,omitempty"`
	GraphDriver           string                    `json:"storage-driver,omitempty"`
	GraphOptions          []string                  `json:"storage-opts,omitempty"`
	Labels                []string                  `json:"labels,omitempty"`
	Mtu                   int                       `json:"mtu,omitempty"`
	NetworkDiagnosticPort int                       `json:"network-diagnostic-port,omitempty"`
	Pidfile               string                    `json:"pidfile,omitempty"`
	RawLogs               bool                      `json:"raw-logs,omitempty"`
	RootDeprecated        string                    `json:"graph,omitempty"`
	Root                  string                    `json:"data-root,omitempty"`
	ExecRoot              string                    `json:"exec-root,omitempty"`
	SocketGroup           string                    `json:"group,omitempty"`
	CorsHeaders           string                    `json:"api-cors-header,omitempty"`

	// TrustKeyPath is used to generate the daemon ID and for signing schema 1 manifests
	// when pushing to a registry which does not support schema 2. This field is marked as
	// deprecated because schema 1 manifests are deprecated in favor of schema 2 and the
	// daemon ID will use a dedicated identifier not shared with exported signatures.
	TrustKeyPath string `json:"deprecated-key-path,omitempty"`

	// LiveRestoreEnabled determines whether we should keep containers
	// alive upon daemon shutdown/start
	LiveRestoreEnabled bool `json:"live-restore,omitempty"`

	// ClusterStore is the storage backend used for the cluster information. It is used by both
	// multihost networking (to store networks and endpoints information) and by the node discovery
	// mechanism.
	ClusterStore string `json:"cluster-store,omitempty"`

	// ClusterOpts is used to pass options to the discovery package for tuning libkv settings, such
	// as TLS configuration settings.
	ClusterOpts map[string]string `json:"cluster-store-opts,omitempty"`

	// ClusterAdvertise is the network endpoint that the Engine advertises for the purpose of node
	// discovery. This should be a 'host:port' combination on which that daemon instance is
	// reachable by other hosts.
	ClusterAdvertise string `json:"cluster-advertise,omitempty"`

	// MaxConcurrentDownloads is the maximum number of downloads that
	// may take place at a time for each pull.
	MaxConcurrentDownloads *int `json:"max-concurrent-downloads,omitempty"`

	// MaxConcurrentUploads is the maximum number of uploads that
	// may take place at a time for each push.
	MaxConcurrentUploads *int `json:"max-concurrent-uploads,omitempty"`

	// ShutdownTimeout is the timeout value (in seconds) the daemon will wait for the container
	// to stop when daemon is being shutdown
	ShutdownTimeout int `json:"shutdown-timeout,omitempty"`

	Debug     bool     `json:"debug,omitempty"`
	Hosts     []string `json:"hosts,omitempty"`
	LogLevel  string   `json:"log-level,omitempty"`
	TLS       bool     `json:"tls,omitempty"`
	TLSVerify bool     `json:"tlsverify,omitempty"`

	// Embedded structs that allow config
	// deserialization without the full struct.
	CommonTLSOptions

	// SwarmDefaultAdvertiseAddr is the default host/IP or network interface
	// to use if a wildcard address is specified in the ListenAddr value
	// given to the /swarm/init endpoint and no advertise address is
	// specified.
	SwarmDefaultAdvertiseAddr string `json:"swarm-default-advertise-addr"`
	MetricsAddress            string `json:"metrics-addr"`

	LogConfig
	BridgeConfig // bridgeConfig holds bridge network specific configuration.
	registry.ServiceOptions

	sync.Mutex
	// FIXME(vdemeester) This part is not that clear and is mainly dependent on cli flags
	// It should probably be handled outside this package.
	ValuesSet map[string]interface{}

	Experimental bool `json:"experimental"` // Experimental indicates whether experimental features should be exposed or not

	// Exposed node Generic Resources
	// e.g: ["orange=red", "orange=green", "orange=blue", "apple=3"]
	NodeGenericResources []string `json:"node-generic-resources,omitempty"`
	// NetworkControlPlaneMTU allows to specify the control plane MTU, this will allow to optimize the network use in some components
	NetworkControlPlaneMTU int `json:"network-control-plane-mtu,omitempty"`

	// ContainerAddr is the address used to connect to containerd if we're
	// not starting it ourselves
	ContainerdAddr string `json:"containerd,omitempty"`
}

// IsValueSet returns true if a configuration value
// was explicitly set in the configuration file.
func (conf *Config) IsValueSet(name string) bool {
	if conf.ValuesSet == nil {
		return false
	}
	_, ok := conf.ValuesSet[name]
	return ok
}

// New returns a new fully initialized Config struct
func New() *Config {
	config := Config{}
	config.LogConfig.Config = make(map[string]string)
	config.ClusterOpts = make(map[string]string)

	if runtime.GOOS != "linux" {
		config.V2Only = true
	}
	return &config
}

// ParseClusterAdvertiseSettings parses the specified advertise settings
func ParseClusterAdvertiseSettings(clusterStore, clusterAdvertise string) (string, error) {
	if clusterAdvertise == "" {
		return "", daemondiscovery.ErrDiscoveryDisabled
	}
	if clusterStore == "" {
		return "", errors.New("invalid cluster configuration. --cluster-advertise must be accompanied by --cluster-store configuration")
	}

	advertise, err := discovery.ParseAdvertise(clusterAdvertise)
	if err != nil {
		return "", fmt.Errorf("discovery advertise parsing failed (%v)", err)
	}
	return advertise, nil
}

// GetConflictFreeLabels validates Labels for conflict
// In swarm the duplicates for labels are removed
// so we only take same values here, no conflict values
// If the key-value is the same we will only take the last label
func GetConflictFreeLabels(labels []string) ([]string, error) {
	labelMap := map[string]string{}
	for _, label := range labels {
		stringSlice := strings.SplitN(label, "=", 2)
		if len(stringSlice) > 1 {
			// If there is a conflict we will return an error
			if v, ok := labelMap[stringSlice[0]]; ok && v != stringSlice[1] {
				return nil, fmt.Errorf("conflict labels for %s=%s and %s=%s", stringSlice[0], stringSlice[1], stringSlice[0], v)
			}
			labelMap[stringSlice[0]] = stringSlice[1]
		}
	}

	newLabels := []string{}
	for k, v := range labelMap {
		newLabels = append(newLabels, fmt.Sprintf("%s=%s", k, v))
	}
	return newLabels, nil
}

// Reload reads the configuration in the host and reloads the daemon and server.
func Reload(configFile string, flags *pflag.FlagSet, reload func(*Config)) error {
	logrus.Infof("Got signal to reload configuration, reloading from: %s", configFile)
	newConfig, err := getConflictFreeConfiguration(configFile, flags)
	if err != nil {
		if flags.Changed("config-file") || !os.IsNotExist(err) {
			return fmt.Errorf("unable to configure the Docker daemon with file %s: %v", configFile, err)
		}
		newConfig = New()
	}

	if err := Validate(newConfig); err != nil {
		return fmt.Errorf("file configuration validation failed (%v)", err)
	}

	// Check if duplicate label-keys with different values are found
	newLabels, err := GetConflictFreeLabels(newConfig.Labels)
	if err != nil {
		return err
	}
	newConfig.Labels = newLabels

	reload(newConfig)
	return nil
}

// boolValue is an interface that boolean value flags implement
// to tell the command line how to make -name equivalent to -name=true.
type boolValue interface {
	IsBoolFlag() bool
}

// MergeDaemonConfigurations reads a configuration file,
// loads the file configuration in an isolated structure,
// and merges the configuration provided from flags on top
// if there are no conflicts.
func MergeDaemonConfigurations(flagsConfig *Config, flags *pflag.FlagSet, configFile string) (*Config, error) {
	fileConfig, err := getConflictFreeConfiguration(configFile, flags)
	if err != nil {
		return nil, err
	}

	if err := Validate(fileConfig); err != nil {
		return nil, fmt.Errorf("configuration validation from file failed (%v)", err)
	}

	// merge flags configuration on top of the file configuration
	if err := mergo.Merge(fileConfig, flagsConfig); err != nil {
		return nil, err
	}

	// We need to validate again once both fileConfig and flagsConfig
	// have been merged
	if err := Validate(fileConfig); err != nil {
		return nil, fmt.Errorf("merged configuration validation from file and command line flags failed (%v)", err)
	}

	return fileConfig, nil
}

// getConflictFreeConfiguration loads the configuration from a JSON file.
// It compares that configuration with the one provided by the flags,
// and returns an error if there are conflicts.
func getConflictFreeConfiguration(configFile string, flags *pflag.FlagSet) (*Config, error) {
	b, err := ioutil.ReadFile(configFile)
	if err != nil {
		return nil, err
	}

	var config Config
	var reader io.Reader
	if flags != nil {
		var jsonConfig map[string]interface{}
		reader = bytes.NewReader(b)
		if err := json.NewDecoder(reader).Decode(&jsonConfig); err != nil {
			return nil, err
		}

		configSet := configValuesSet(jsonConfig)

		if err := findConfigurationConflicts(configSet, flags); err != nil {
			return nil, err
		}

		// Override flag values to make sure the values set in the config file with nullable values, like `false`,
		// are not overridden by default truthy values from the flags that were not explicitly set.
		// See https://github.com/docker/docker/issues/20289 for an example.
		//
		// TODO: Rewrite configuration logic to avoid same issue with other nullable values, like numbers.
		namedOptions := make(map[string]interface{})
		for key, value := range configSet {
			f := flags.Lookup(key)
			if f == nil { // ignore named flags that don't match
				namedOptions[key] = value
				continue
			}

			if _, ok := f.Value.(boolValue); ok {
				f.Value.Set(fmt.Sprintf("%v", value))
			}
		}
		if len(namedOptions) > 0 {
			// set also default for mergeVal flags that are boolValue at the same time.
			flags.VisitAll(func(f *pflag.Flag) {
				if opt, named := f.Value.(opts.NamedOption); named {
					v, set := namedOptions[opt.Name()]
					_, boolean := f.Value.(boolValue)
					if set && boolean {
						f.Value.Set(fmt.Sprintf("%v", v))
					}
				}
			})
		}

		config.ValuesSet = configSet
	}

	reader = bytes.NewReader(b)
	if err := json.NewDecoder(reader).Decode(&config); err != nil {
		return nil, err
	}

	if config.RootDeprecated != "" {
		logrus.Warn(`The "graph" config file option is deprecated. Please use "data-root" instead.`)

		if config.Root != "" {
			return nil, fmt.Errorf(`cannot specify both "graph" and "data-root" config file options`)
		}

		config.Root = config.RootDeprecated
	}

	return &config, nil
}

// configValuesSet returns the configuration values explicitly set in the file.
func configValuesSet(config map[string]interface{}) map[string]interface{} {
	flatten := make(map[string]interface{})
	for k, v := range config {
		if m, isMap := v.(map[string]interface{}); isMap && !flatOptions[k] {
			for km, vm := range m {
				flatten[km] = vm
			}
			continue
		}

		flatten[k] = v
	}
	return flatten
}

// findConfigurationConflicts iterates over the provided flags searching for
// duplicated configurations and unknown keys. It returns an error with all the conflicts if
// it finds any.
func findConfigurationConflicts(config map[string]interface{}, flags *pflag.FlagSet) error {
	// 1. Search keys from the file that we don't recognize as flags.
	unknownKeys := make(map[string]interface{})
	for key, value := range config {
		if flag := flags.Lookup(key); flag == nil {
			unknownKeys[key] = value
		}
	}

	// 2. Discard values that implement NamedOption.
	// Their configuration name differs from their flag name, like `labels` and `label`.
	if len(unknownKeys) > 0 {
		unknownNamedConflicts := func(f *pflag.Flag) {
			if namedOption, ok := f.Value.(opts.NamedOption); ok {
				if _, valid := unknownKeys[namedOption.Name()]; valid {
					delete(unknownKeys, namedOption.Name())
				}
			}
		}
		flags.VisitAll(unknownNamedConflicts)
	}

	if len(unknownKeys) > 0 {
		var unknown []string
		for key := range unknownKeys {
			unknown = append(unknown, key)
		}
		return fmt.Errorf("the following directives don't match any configuration option: %s", strings.Join(unknown, ", "))
	}

	var conflicts []string
	printConflict := func(name string, flagValue, fileValue interface{}) string {
		return fmt.Sprintf("%s: (from flag: %v, from file: %v)", name, flagValue, fileValue)
	}

	// 3. Search keys that are present as a flag and as a file option.
	duplicatedConflicts := func(f *pflag.Flag) {
		// search option name in the json configuration payload if the value is a named option
		if namedOption, ok := f.Value.(opts.NamedOption); ok {
			if optsValue, ok := config[namedOption.Name()]; ok {
				conflicts = append(conflicts, printConflict(namedOption.Name(), f.Value.String(), optsValue))
			}
		} else {
			// search flag name in the json configuration payload
			for _, name := range []string{f.Name, f.Shorthand} {
				if value, ok := config[name]; ok {
					conflicts = append(conflicts, printConflict(name, f.Value.String(), value))
					break
				}
			}
		}
	}

	flags.Visit(duplicatedConflicts)

	if len(conflicts) > 0 {
		return fmt.Errorf("the following directives are specified both as a flag and in the configuration file: %s", strings.Join(conflicts, ", "))
	}
	return nil
}

// Validate validates some specific configs.
// such as config.DNS, config.Labels, config.DNSSearch,
// as well as config.MaxConcurrentDownloads, config.MaxConcurrentUploads.
func Validate(config *Config) error {
	// validate DNS
	for _, dns := range config.DNS {
		if _, err := opts.ValidateIPAddress(dns); err != nil {
			return err
		}
	}

	// validate DNSSearch
	for _, dnsSearch := range config.DNSSearch {
		if _, err := opts.ValidateDNSSearch(dnsSearch); err != nil {
			return err
		}
	}

	// validate Labels
	for _, label := range config.Labels {
		if _, err := opts.ValidateLabel(label); err != nil {
			return err
		}
	}
	// validate MaxConcurrentDownloads
	if config.MaxConcurrentDownloads != nil && *config.MaxConcurrentDownloads < 0 {
		return fmt.Errorf("invalid max concurrent downloads: %d", *config.MaxConcurrentDownloads)
	}
	// validate MaxConcurrentUploads
	if config.MaxConcurrentUploads != nil && *config.MaxConcurrentUploads < 0 {
		return fmt.Errorf("invalid max concurrent uploads: %d", *config.MaxConcurrentUploads)
	}

	// validate that "default" runtime is not reset
	if runtimes := config.GetAllRuntimes(); len(runtimes) > 0 {
		if _, ok := runtimes[StockRuntimeName]; ok {
			return fmt.Errorf("runtime name '%s' is reserved", StockRuntimeName)
		}
	}

	if _, err := ParseGenericResources(config.NodeGenericResources); err != nil {
		return err
	}

	if defaultRuntime := config.GetDefaultRuntimeName(); defaultRuntime != "" && defaultRuntime != StockRuntimeName {
		runtimes := config.GetAllRuntimes()
		if _, ok := runtimes[defaultRuntime]; !ok {
			return fmt.Errorf("specified default runtime '%s' does not exist", defaultRuntime)
		}
	}

	// validate platform-specific settings
	if err := config.ValidatePlatformConfig(); err != nil {
		return err
	}

	return nil
}

// ModifiedDiscoverySettings returns whether the discovery configuration has been modified or not.
func ModifiedDiscoverySettings(config *Config, backendType, advertise string, clusterOpts map[string]string) bool {
	if config.ClusterStore != backendType || config.ClusterAdvertise != advertise {
		return true
	}

	if (config.ClusterOpts == nil && clusterOpts == nil) ||
		(config.ClusterOpts == nil && len(clusterOpts) == 0) ||
		(len(config.ClusterOpts) == 0 && clusterOpts == nil) {
		return false
	}

	return !reflect.DeepEqual(config.ClusterOpts, clusterOpts)
}
