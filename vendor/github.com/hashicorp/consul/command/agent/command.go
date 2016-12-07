package agent

import (
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/armon/go-metrics"
	"github.com/armon/go-metrics/circonus"
	"github.com/armon/go-metrics/datadog"
	"github.com/hashicorp/consul/lib"
	"github.com/hashicorp/consul/watch"
	"github.com/hashicorp/go-checkpoint"
	"github.com/hashicorp/go-reap"
	"github.com/hashicorp/go-syslog"
	"github.com/hashicorp/logutils"
	scada "github.com/hashicorp/scada-client/scada"
	"github.com/mitchellh/cli"
)

// gracefulTimeout controls how long we wait before forcefully terminating
var gracefulTimeout = 5 * time.Second

// validDatacenter is used to validate a datacenter
var validDatacenter = regexp.MustCompile("^[a-zA-Z0-9_-]+$")

// Command is a Command implementation that runs a Consul agent.
// The command will not end unless a shutdown message is sent on the
// ShutdownCh. If two messages are sent on the ShutdownCh it will forcibly
// exit.
type Command struct {
	Revision          string
	Version           string
	VersionPrerelease string
	HumanVersion      string
	Ui                cli.Ui
	ShutdownCh        <-chan struct{}
	args              []string
	logFilter         *logutils.LevelFilter
	logOutput         io.Writer
	agent             *Agent
	rpcServer         *AgentRPC
	httpServers       []*HTTPServer
	dnsServer         *DNSServer
	scadaProvider     *scada.Provider
	scadaHttp         *HTTPServer
}

// readConfig is responsible for setup of our configuration using
// the command line and any file configs
func (c *Command) readConfig() *Config {
	var cmdConfig Config
	var configFiles []string
	var retryInterval string
	var retryIntervalWan string
	var dnsRecursors []string
	var dev bool
	var dcDeprecated string
	cmdFlags := flag.NewFlagSet("agent", flag.ContinueOnError)
	cmdFlags.Usage = func() { c.Ui.Output(c.Help()) }

	cmdFlags.Var((*AppendSliceValue)(&configFiles), "config-file", "json file to read config from")
	cmdFlags.Var((*AppendSliceValue)(&configFiles), "config-dir", "directory of json files to read")
	cmdFlags.Var((*AppendSliceValue)(&dnsRecursors), "recursor", "address of an upstream DNS server")
	cmdFlags.BoolVar(&dev, "dev", false, "development server mode")

	cmdFlags.StringVar(&cmdConfig.LogLevel, "log-level", "", "log level")
	cmdFlags.StringVar(&cmdConfig.NodeName, "node", "", "node name")
	cmdFlags.StringVar(&dcDeprecated, "dc", "", "node datacenter (deprecated: use 'datacenter' instead)")
	cmdFlags.StringVar(&cmdConfig.Datacenter, "datacenter", "", "node datacenter")
	cmdFlags.StringVar(&cmdConfig.DataDir, "data-dir", "", "path to the data directory")
	cmdFlags.BoolVar(&cmdConfig.EnableUi, "ui", false, "enable the built-in web UI")
	cmdFlags.StringVar(&cmdConfig.UiDir, "ui-dir", "", "path to the web UI directory")
	cmdFlags.StringVar(&cmdConfig.PidFile, "pid-file", "", "path to file to store PID")
	cmdFlags.StringVar(&cmdConfig.EncryptKey, "encrypt", "", "gossip encryption key")

	cmdFlags.BoolVar(&cmdConfig.Server, "server", false, "run agent as server")
	cmdFlags.BoolVar(&cmdConfig.Bootstrap, "bootstrap", false, "enable server bootstrap mode")
	cmdFlags.IntVar(&cmdConfig.BootstrapExpect, "bootstrap-expect", 0, "enable automatic bootstrap via expect mode")
	cmdFlags.StringVar(&cmdConfig.Domain, "domain", "", "domain to use for DNS interface")

	cmdFlags.StringVar(&cmdConfig.ClientAddr, "client", "", "address to bind client listeners to (DNS, HTTP, HTTPS, RPC)")
	cmdFlags.StringVar(&cmdConfig.BindAddr, "bind", "", "address to bind server listeners to")
	cmdFlags.IntVar(&cmdConfig.Ports.HTTP, "http-port", 0, "http port to use")
	cmdFlags.IntVar(&cmdConfig.Ports.DNS, "dns-port", 0, "DNS port to use")
	cmdFlags.StringVar(&cmdConfig.AdvertiseAddr, "advertise", "", "address to advertise instead of bind addr")
	cmdFlags.StringVar(&cmdConfig.AdvertiseAddrWan, "advertise-wan", "", "address to advertise on wan instead of bind or advertise addr")

	cmdFlags.StringVar(&cmdConfig.AtlasInfrastructure, "atlas", "", "infrastructure name in Atlas")
	cmdFlags.StringVar(&cmdConfig.AtlasToken, "atlas-token", "", "authentication token for Atlas")
	cmdFlags.BoolVar(&cmdConfig.AtlasJoin, "atlas-join", false, "auto-join with Atlas")
	cmdFlags.StringVar(&cmdConfig.AtlasEndpoint, "atlas-endpoint", "", "endpoint for Atlas integration")

	cmdFlags.IntVar(&cmdConfig.Protocol, "protocol", -1, "protocol version")

	cmdFlags.BoolVar(&cmdConfig.EnableSyslog, "syslog", false,
		"enable logging to syslog facility")
	cmdFlags.BoolVar(&cmdConfig.RejoinAfterLeave, "rejoin", false,
		"enable re-joining after a previous leave")
	cmdFlags.Var((*AppendSliceValue)(&cmdConfig.StartJoin), "join",
		"address of agent to join on startup")
	cmdFlags.Var((*AppendSliceValue)(&cmdConfig.StartJoinWan), "join-wan",
		"address of agent to join -wan on startup")
	cmdFlags.Var((*AppendSliceValue)(&cmdConfig.RetryJoin), "retry-join",
		"address of agent to join on startup with retry")
	cmdFlags.IntVar(&cmdConfig.RetryMaxAttempts, "retry-max", 0,
		"number of retries for joining")
	cmdFlags.StringVar(&retryInterval, "retry-interval", "",
		"interval between join attempts")
	cmdFlags.Var((*AppendSliceValue)(&cmdConfig.RetryJoinWan), "retry-join-wan",
		"address of agent to join -wan on startup with retry")
	cmdFlags.IntVar(&cmdConfig.RetryMaxAttemptsWan, "retry-max-wan", 0,
		"number of retries for joining -wan")
	cmdFlags.StringVar(&retryIntervalWan, "retry-interval-wan", "",
		"interval between join -wan attempts")

	if err := cmdFlags.Parse(c.args); err != nil {
		return nil
	}

	if retryInterval != "" {
		dur, err := time.ParseDuration(retryInterval)
		if err != nil {
			c.Ui.Error(fmt.Sprintf("Error: %s", err))
			return nil
		}
		cmdConfig.RetryInterval = dur
	}

	if retryIntervalWan != "" {
		dur, err := time.ParseDuration(retryIntervalWan)
		if err != nil {
			c.Ui.Error(fmt.Sprintf("Error: %s", err))
			return nil
		}
		cmdConfig.RetryIntervalWan = dur
	}

	var config *Config
	if dev {
		config = DevConfig()
	} else {
		config = DefaultConfig()
	}

	if len(configFiles) > 0 {
		fileConfig, err := ReadConfigPaths(configFiles)
		if err != nil {
			c.Ui.Error(err.Error())
			return nil
		}

		config = MergeConfig(config, fileConfig)
	}

	cmdConfig.DNSRecursors = append(cmdConfig.DNSRecursors, dnsRecursors...)

	config = MergeConfig(config, &cmdConfig)

	if config.NodeName == "" {
		hostname, err := os.Hostname()
		if err != nil {
			c.Ui.Error(fmt.Sprintf("Error determining node name: %s", err))
			return nil
		}
		config.NodeName = hostname
	}
	config.NodeName = strings.TrimSpace(config.NodeName)
	if config.NodeName == "" {
		c.Ui.Error("Node name can not be empty")
		return nil
	}

	// Make sure LeaveOnTerm and SkipLeaveOnInt are set to the right
	// defaults based on the agent's mode (client or server).
	if config.LeaveOnTerm == nil {
		config.LeaveOnTerm = Bool(!config.Server)
	}
	if config.SkipLeaveOnInt == nil {
		config.SkipLeaveOnInt = Bool(config.Server)
	}

	// Ensure we have a data directory
	if config.DataDir == "" && !dev {
		c.Ui.Error("Must specify data directory using -data-dir")
		return nil
	}

	// Ensure all endpoints are unique
	if err := config.verifyUniqueListeners(); err != nil {
		c.Ui.Error(fmt.Sprintf("All listening endpoints must be unique: %s", err))
		return nil
	}

	// Check the data dir for signs of an un-migrated Consul 0.5.x or older
	// server. Consul refuses to start if this is present to protect a server
	// with existing data from starting on a fresh data set.
	if config.Server {
		mdbPath := filepath.Join(config.DataDir, "mdb")
		if _, err := os.Stat(mdbPath); !os.IsNotExist(err) {
			if os.IsPermission(err) {
				c.Ui.Error(fmt.Sprintf("CRITICAL: Permission denied for data folder at %q!", mdbPath))
				c.Ui.Error("Consul will refuse to boot without access to this directory.")
				c.Ui.Error("Please correct permissions and try starting again.")
				return nil
			}
			c.Ui.Error(fmt.Sprintf("CRITICAL: Deprecated data folder found at %q!", mdbPath))
			c.Ui.Error("Consul will refuse to boot with this directory present.")
			c.Ui.Error("See https://www.consul.io/docs/upgrade-specific.html for more information.")
			return nil
		}
	}

	// Verify DNS settings
	if config.DNSConfig.UDPAnswerLimit < 1 {
		c.Ui.Error(fmt.Sprintf("dns_config.udp_answer_limit %d too low, must always be greater than zero", config.DNSConfig.UDPAnswerLimit))
	}

	if config.EncryptKey != "" {
		if _, err := config.EncryptBytes(); err != nil {
			c.Ui.Error(fmt.Sprintf("Invalid encryption key: %s", err))
			return nil
		}
		keyfileLAN := filepath.Join(config.DataDir, serfLANKeyring)
		if _, err := os.Stat(keyfileLAN); err == nil {
			c.Ui.Error("WARNING: LAN keyring exists but -encrypt given, using keyring")
		}
		if config.Server {
			keyfileWAN := filepath.Join(config.DataDir, serfWANKeyring)
			if _, err := os.Stat(keyfileWAN); err == nil {
				c.Ui.Error("WARNING: WAN keyring exists but -encrypt given, using keyring")
			}
		}
	}

	// Output a warning if the 'dc' flag has been used.
	if dcDeprecated != "" {
		c.Ui.Error("WARNING: the 'dc' flag has been deprecated. Use 'datacenter' instead")

		// Making sure that we don't break previous versions.
		config.Datacenter = dcDeprecated
	}

	// Ensure the datacenter is always lowercased. The DNS endpoints automatically
	// lowercase all queries, and internally we expect DC1 and dc1 to be the same.
	config.Datacenter = strings.ToLower(config.Datacenter)

	// Verify datacenter is valid
	if !validDatacenter.MatchString(config.Datacenter) {
		c.Ui.Error("Datacenter must be alpha-numeric with underscores and hypens only")
		return nil
	}

	// Only allow bootstrap mode when acting as a server
	if config.Bootstrap && !config.Server {
		c.Ui.Error("Bootstrap mode cannot be enabled when server mode is not enabled")
		return nil
	}

	// Expect can only work when acting as a server
	if config.BootstrapExpect != 0 && !config.Server {
		c.Ui.Error("Expect mode cannot be enabled when server mode is not enabled")
		return nil
	}

	// Expect & Bootstrap are mutually exclusive
	if config.BootstrapExpect != 0 && config.Bootstrap {
		c.Ui.Error("Bootstrap cannot be provided with an expected server count")
		return nil
	}

	// Compile all the watches
	for _, params := range config.Watches {
		// Parse the watches, excluding the handler
		wp, err := watch.ParseExempt(params, []string{"handler"})
		if err != nil {
			c.Ui.Error(fmt.Sprintf("Failed to parse watch (%#v): %v", params, err))
			return nil
		}

		// Get the handler
		if err := verifyWatchHandler(wp.Exempt["handler"]); err != nil {
			c.Ui.Error(fmt.Sprintf("Failed to setup watch handler (%#v): %v", params, err))
			return nil
		}

		// Store the watch plan
		config.WatchPlans = append(config.WatchPlans, wp)
	}

	// Warn if we are in expect mode
	if config.BootstrapExpect == 1 {
		c.Ui.Error("WARNING: BootstrapExpect Mode is specified as 1; this is the same as Bootstrap mode.")
		config.BootstrapExpect = 0
		config.Bootstrap = true
	} else if config.BootstrapExpect > 0 {
		c.Ui.Error(fmt.Sprintf("WARNING: Expect Mode enabled, expecting %d servers", config.BootstrapExpect))
	}

	// Warn if we are in bootstrap mode
	if config.Bootstrap {
		c.Ui.Error("WARNING: Bootstrap mode enabled! Do not enable unless necessary")
	}

	// Set the version info
	config.Revision = c.Revision
	config.Version = c.Version
	config.VersionPrerelease = c.VersionPrerelease

	return config
}

// verifyUniqueListeners checks to see if an address was used more than once in
// the config
func (config *Config) verifyUniqueListeners() error {
	listeners := []struct {
		host  string
		port  int
		descr string
	}{
		{config.Addresses.RPC, config.Ports.RPC, "RPC"},
		{config.Addresses.DNS, config.Ports.DNS, "DNS"},
		{config.Addresses.HTTP, config.Ports.HTTP, "HTTP"},
		{config.Addresses.HTTPS, config.Ports.HTTPS, "HTTPS"},
		{config.AdvertiseAddr, config.Ports.Server, "Server RPC"},
		{config.AdvertiseAddr, config.Ports.SerfLan, "Serf LAN"},
		{config.AdvertiseAddr, config.Ports.SerfWan, "Serf WAN"},
	}

	type key struct {
		host string
		port int
	}
	m := make(map[key]string, len(listeners))

	for _, l := range listeners {
		if l.host == "" {
			l.host = "0.0.0.0"
		} else if strings.HasPrefix(l.host, "unix") {
			// Don't compare ports on unix sockets
			l.port = 0
		}
		if l.host == "0.0.0.0" && l.port <= 0 {
			continue
		}

		k := key{l.host, l.port}
		v, ok := m[k]
		if ok {
			return fmt.Errorf("%s address already configured for %s", l.descr, v)
		}
		m[k] = l.descr
	}
	return nil
}

// setupLoggers is used to setup the logGate, logWriter, and our logOutput
func (c *Command) setupLoggers(config *Config) (*GatedWriter, *logWriter, io.Writer) {
	// Setup logging. First create the gated log writer, which will
	// store logs until we're ready to show them. Then create the level
	// filter, filtering logs of the specified level.
	logGate := &GatedWriter{
		Writer: &cli.UiWriter{Ui: c.Ui},
	}

	c.logFilter = LevelFilter()
	c.logFilter.MinLevel = logutils.LogLevel(strings.ToUpper(config.LogLevel))
	c.logFilter.Writer = logGate
	if !ValidateLevelFilter(c.logFilter.MinLevel, c.logFilter) {
		c.Ui.Error(fmt.Sprintf(
			"Invalid log level: %s. Valid log levels are: %v",
			c.logFilter.MinLevel, c.logFilter.Levels))
		return nil, nil, nil
	}

	// Check if syslog is enabled
	var syslog io.Writer
	if config.EnableSyslog {
		l, err := gsyslog.NewLogger(gsyslog.LOG_NOTICE, config.SyslogFacility, "consul")
		if err != nil {
			c.Ui.Error(fmt.Sprintf("Syslog setup failed: %v", err))
			return nil, nil, nil
		}
		syslog = &SyslogWrapper{l, c.logFilter}
	}

	// Create a log writer, and wrap a logOutput around it
	logWriter := NewLogWriter(512)
	var logOutput io.Writer
	if syslog != nil {
		logOutput = io.MultiWriter(c.logFilter, logWriter, syslog)
	} else {
		logOutput = io.MultiWriter(c.logFilter, logWriter)
	}
	c.logOutput = logOutput
	return logGate, logWriter, logOutput
}

// setupAgent is used to start the agent and various interfaces
func (c *Command) setupAgent(config *Config, logOutput io.Writer, logWriter *logWriter) error {
	c.Ui.Output("Starting Consul agent...")
	agent, err := Create(config, logOutput)
	if err != nil {
		c.Ui.Error(fmt.Sprintf("Error starting agent: %s", err))
		return err
	}
	c.agent = agent

	// Setup the RPC listener
	rpcAddr, err := config.ClientListener(config.Addresses.RPC, config.Ports.RPC)
	if err != nil {
		c.Ui.Error(fmt.Sprintf("Invalid RPC bind address: %s", err))
		return err
	}

	// Clear the domain socket file if it exists
	socketPath, isSocket := unixSocketAddr(config.Addresses.RPC)
	if isSocket {
		if _, err := os.Stat(socketPath); !os.IsNotExist(err) {
			agent.logger.Printf("[WARN] agent: Replacing socket %q", socketPath)
		}
		if err := os.Remove(socketPath); err != nil && !os.IsNotExist(err) {
			c.Ui.Output(fmt.Sprintf("Error removing socket file: %s", err))
			return err
		}
	}

	rpcListener, err := net.Listen(rpcAddr.Network(), rpcAddr.String())
	if err != nil {
		agent.Shutdown()
		c.Ui.Error(fmt.Sprintf("Error starting RPC listener: %s", err))
		return err
	}

	// Set up ownership/permission bits on the socket file
	if isSocket {
		if err := setFilePermissions(socketPath, config.UnixSockets); err != nil {
			agent.Shutdown()
			c.Ui.Error(fmt.Sprintf("Error setting up socket: %s", err))
			return err
		}
	}

	// Start the IPC layer
	c.Ui.Output("Starting Consul agent RPC...")
	c.rpcServer = NewAgentRPC(agent, rpcListener, logOutput, logWriter)

	// Enable the SCADA integration
	if err := c.setupScadaConn(config); err != nil {
		agent.Shutdown()
		c.Ui.Error(fmt.Sprintf("Error starting SCADA connection: %s", err))
		return err
	}

	if config.Ports.HTTP > 0 || config.Ports.HTTPS > 0 {
		servers, err := NewHTTPServers(agent, config, logOutput)
		if err != nil {
			agent.Shutdown()
			c.Ui.Error(fmt.Sprintf("Error starting http servers: %s", err))
			return err
		}
		c.httpServers = servers
	}

	if config.Ports.DNS > 0 {
		dnsAddr, err := config.ClientListener(config.Addresses.DNS, config.Ports.DNS)
		if err != nil {
			agent.Shutdown()
			c.Ui.Error(fmt.Sprintf("Invalid DNS bind address: %s", err))
			return err
		}

		server, err := NewDNSServer(agent, &config.DNSConfig, logOutput,
			config.Domain, dnsAddr.String(), config.DNSRecursors)
		if err != nil {
			agent.Shutdown()
			c.Ui.Error(fmt.Sprintf("Error starting dns server: %s", err))
			return err
		}
		c.dnsServer = server
	}

	// Setup update checking
	if !config.DisableUpdateCheck {
		version := config.Version
		if config.VersionPrerelease != "" {
			version += fmt.Sprintf("-%s", config.VersionPrerelease)
		}
		updateParams := &checkpoint.CheckParams{
			Product: "consul",
			Version: version,
		}
		if !config.DisableAnonymousSignature {
			updateParams.SignatureFile = filepath.Join(config.DataDir, "checkpoint-signature")
		}

		// Schedule a periodic check with expected interval of 24 hours
		checkpoint.CheckInterval(updateParams, 24*time.Hour, c.checkpointResults)

		// Do an immediate check within the next 30 seconds
		go func() {
			time.Sleep(lib.RandomStagger(30 * time.Second))
			c.checkpointResults(checkpoint.Check(updateParams))
		}()
	}
	return nil
}

// checkpointResults is used to handler periodic results from our update checker
func (c *Command) checkpointResults(results *checkpoint.CheckResponse, err error) {
	if err != nil {
		c.Ui.Error(fmt.Sprintf("Failed to check for updates: %v", err))
		return
	}
	if results.Outdated {
		c.Ui.Error(fmt.Sprintf("Newer Consul version available: %s (currently running: %s)", results.CurrentVersion, c.Version))
	}
	for _, alert := range results.Alerts {
		switch alert.Level {
		case "info":
			c.Ui.Info(fmt.Sprintf("Bulletin [%s]: %s (%s)", alert.Level, alert.Message, alert.URL))
		default:
			c.Ui.Error(fmt.Sprintf("Bulletin [%s]: %s (%s)", alert.Level, alert.Message, alert.URL))
		}
	}
}

// startupJoin is invoked to handle any joins specified to take place at start time
func (c *Command) startupJoin(config *Config) error {
	if len(config.StartJoin) == 0 {
		return nil
	}

	c.Ui.Output("Joining cluster...")
	n, err := c.agent.JoinLAN(config.StartJoin)
	if err != nil {
		return err
	}

	c.Ui.Info(fmt.Sprintf("Join completed. Synced with %d initial agents", n))
	return nil
}

// startupJoinWan is invoked to handle any joins -wan specified to take place at start time
func (c *Command) startupJoinWan(config *Config) error {
	if len(config.StartJoinWan) == 0 {
		return nil
	}

	c.Ui.Output("Joining -wan cluster...")
	n, err := c.agent.JoinWAN(config.StartJoinWan)
	if err != nil {
		return err
	}

	c.Ui.Info(fmt.Sprintf("Join -wan completed. Synced with %d initial agents", n))
	return nil
}

// retryJoin is used to handle retrying a join until it succeeds or all
// retries are exhausted.
func (c *Command) retryJoin(config *Config, errCh chan<- struct{}) {
	if len(config.RetryJoin) == 0 {
		return
	}

	logger := c.agent.logger
	logger.Printf("[INFO] agent: Joining cluster...")

	attempt := 0
	for {
		n, err := c.agent.JoinLAN(config.RetryJoin)
		if err == nil {
			logger.Printf("[INFO] agent: Join completed. Synced with %d initial agents", n)
			return
		}

		attempt++
		if config.RetryMaxAttempts > 0 && attempt > config.RetryMaxAttempts {
			logger.Printf("[ERROR] agent: max join retry exhausted, exiting")
			close(errCh)
			return
		}

		logger.Printf("[WARN] agent: Join failed: %v, retrying in %v", err,
			config.RetryInterval)
		time.Sleep(config.RetryInterval)
	}
}

// retryJoinWan is used to handle retrying a join -wan until it succeeds or all
// retries are exhausted.
func (c *Command) retryJoinWan(config *Config, errCh chan<- struct{}) {
	if len(config.RetryJoinWan) == 0 {
		return
	}

	logger := c.agent.logger
	logger.Printf("[INFO] agent: Joining WAN cluster...")

	attempt := 0
	for {
		n, err := c.agent.JoinWAN(config.RetryJoinWan)
		if err == nil {
			logger.Printf("[INFO] agent: Join -wan completed. Synced with %d initial agents", n)
			return
		}

		attempt++
		if config.RetryMaxAttemptsWan > 0 && attempt > config.RetryMaxAttemptsWan {
			logger.Printf("[ERROR] agent: max join -wan retry exhausted, exiting")
			close(errCh)
			return
		}

		logger.Printf("[WARN] agent: Join -wan failed: %v, retrying in %v", err,
			config.RetryIntervalWan)
		time.Sleep(config.RetryIntervalWan)
	}
}

// gossipEncrypted determines if the consul instance is using symmetric
// encryption keys to protect gossip protocol messages.
func (c *Command) gossipEncrypted() bool {
	if c.agent.config.EncryptKey != "" {
		return true
	}

	server := c.agent.server
	if server != nil {
		return server.KeyManagerLAN() != nil || server.KeyManagerWAN() != nil
	}

	client := c.agent.client
	return client != nil && client.KeyManagerLAN() != nil
}

func (c *Command) Run(args []string) int {
	c.Ui = &cli.PrefixedUi{
		OutputPrefix: "==> ",
		InfoPrefix:   "    ",
		ErrorPrefix:  "==> ",
		Ui:           c.Ui,
	}

	// Parse our configs
	c.args = args
	config := c.readConfig()
	if config == nil {
		return 1
	}

	// Setup the log outputs
	logGate, logWriter, logOutput := c.setupLoggers(config)
	if logWriter == nil {
		return 1
	}

	/* Setup telemetry
	Aggregate on 10 second intervals for 1 minute. Expose the
	metrics over stderr when there is a SIGUSR1 received.
	*/
	inm := metrics.NewInmemSink(10*time.Second, time.Minute)
	metrics.DefaultInmemSignal(inm)
	metricsConf := metrics.DefaultConfig(config.Telemetry.StatsitePrefix)
	metricsConf.EnableHostname = !config.Telemetry.DisableHostname

	// Configure the statsite sink
	var fanout metrics.FanoutSink
	if config.Telemetry.StatsiteAddr != "" {
		sink, err := metrics.NewStatsiteSink(config.Telemetry.StatsiteAddr)
		if err != nil {
			c.Ui.Error(fmt.Sprintf("Failed to start statsite sink. Got: %s", err))
			return 1
		}
		fanout = append(fanout, sink)
	}

	// Configure the statsd sink
	if config.Telemetry.StatsdAddr != "" {
		sink, err := metrics.NewStatsdSink(config.Telemetry.StatsdAddr)
		if err != nil {
			c.Ui.Error(fmt.Sprintf("Failed to start statsd sink. Got: %s", err))
			return 1
		}
		fanout = append(fanout, sink)
	}

	// Configure the DogStatsd sink
	if config.Telemetry.DogStatsdAddr != "" {
		var tags []string

		if config.Telemetry.DogStatsdTags != nil {
			tags = config.Telemetry.DogStatsdTags
		}

		sink, err := datadog.NewDogStatsdSink(config.Telemetry.DogStatsdAddr, metricsConf.HostName)
		if err != nil {
			c.Ui.Error(fmt.Sprintf("Failed to start DogStatsd sink. Got: %s", err))
			return 1
		}
		sink.SetTags(tags)
		fanout = append(fanout, sink)
	}

	if config.Telemetry.CirconusAPIToken != "" || config.Telemetry.CirconusCheckSubmissionURL != "" {
		cfg := &circonus.Config{}
		cfg.Interval = config.Telemetry.CirconusSubmissionInterval
		cfg.CheckManager.API.TokenKey = config.Telemetry.CirconusAPIToken
		cfg.CheckManager.API.TokenApp = config.Telemetry.CirconusAPIApp
		cfg.CheckManager.API.URL = config.Telemetry.CirconusAPIURL
		cfg.CheckManager.Check.SubmissionURL = config.Telemetry.CirconusCheckSubmissionURL
		cfg.CheckManager.Check.ID = config.Telemetry.CirconusCheckID
		cfg.CheckManager.Check.ForceMetricActivation = config.Telemetry.CirconusCheckForceMetricActivation
		cfg.CheckManager.Check.InstanceID = config.Telemetry.CirconusCheckInstanceID
		cfg.CheckManager.Check.SearchTag = config.Telemetry.CirconusCheckSearchTag
		cfg.CheckManager.Broker.ID = config.Telemetry.CirconusBrokerID
		cfg.CheckManager.Broker.SelectTag = config.Telemetry.CirconusBrokerSelectTag

		if cfg.CheckManager.API.TokenApp == "" {
			cfg.CheckManager.API.TokenApp = "consul"
		}

		if cfg.CheckManager.Check.InstanceID == "" {
			cfg.CheckManager.Check.InstanceID = fmt.Sprintf("%s:%s", config.NodeName, config.Datacenter)
		}

		if cfg.CheckManager.Check.SearchTag == "" {
			cfg.CheckManager.Check.SearchTag = "service:consul"
		}

		sink, err := circonus.NewCirconusSink(cfg)
		if err != nil {
			c.Ui.Error(fmt.Sprintf("Failed to start Circonus sink. Got: %s", err))
			return 1
		}
		sink.Start()
		fanout = append(fanout, sink)
	}

	// Initialize the global sink
	if len(fanout) > 0 {
		fanout = append(fanout, inm)
		metrics.NewGlobal(metricsConf, fanout)
	} else {
		metricsConf.EnableHostname = false
		metrics.NewGlobal(metricsConf, inm)
	}

	// Create the agent
	if err := c.setupAgent(config, logOutput, logWriter); err != nil {
		return 1
	}
	defer c.agent.Shutdown()
	if c.rpcServer != nil {
		defer c.rpcServer.Shutdown()
	}
	if c.dnsServer != nil {
		defer c.dnsServer.Shutdown()
	}
	for _, server := range c.httpServers {
		defer server.Shutdown()
	}

	// Enable child process reaping
	if (config.Reap != nil && *config.Reap) || (config.Reap == nil && os.Getpid() == 1) {
		if !reap.IsSupported() {
			c.Ui.Error("Child process reaping is not supported on this platform (set reap=false)")
			return 1
		} else {
			logger := c.agent.logger
			logger.Printf("[DEBUG] Automatically reaping child processes")

			pids := make(reap.PidCh, 1)
			errors := make(reap.ErrorCh, 1)
			go func() {
				for {
					select {
					case pid := <-pids:
						logger.Printf("[DEBUG] Reaped child process %d", pid)
					case err := <-errors:
						logger.Printf("[ERR] Error reaping child process: %v", err)
					case <-c.agent.shutdownCh:
						return
					}
				}
			}()
			go reap.ReapChildren(pids, errors, c.agent.shutdownCh, &c.agent.reapLock)
		}
	}

	// Check and shut down the SCADA listeners at the end
	defer func() {
		if c.scadaHttp != nil {
			c.scadaHttp.Shutdown()
		}
		if c.scadaProvider != nil {
			c.scadaProvider.Shutdown()
		}
	}()

	// Join startup nodes if specified
	if err := c.startupJoin(config); err != nil {
		c.Ui.Error(err.Error())
		return 1
	}

	// Join startup nodes if specified
	if err := c.startupJoinWan(config); err != nil {
		c.Ui.Error(err.Error())
		return 1
	}

	// Get the new client http listener addr
	httpAddr, err := config.ClientListener(config.Addresses.HTTP, config.Ports.HTTP)
	if err != nil {
		c.Ui.Error(fmt.Sprintf("Failed to determine HTTP address: %v", err))
	}

	// Register the watches
	for _, wp := range config.WatchPlans {
		go func(wp *watch.WatchPlan) {
			wp.Handler = makeWatchHandler(logOutput, wp.Exempt["handler"], &c.agent.reapLock)
			wp.LogOutput = c.logOutput
			if err := wp.Run(httpAddr.String()); err != nil {
				c.Ui.Error(fmt.Sprintf("Error running watch: %v", err))
			}
		}(wp)
	}

	// Figure out if gossip is encrypted
	var gossipEncrypted bool
	if config.Server {
		gossipEncrypted = c.agent.server.Encrypted()
	} else {
		gossipEncrypted = c.agent.client.Encrypted()
	}

	// Determine the Atlas cluster
	atlas := "<disabled>"
	if config.AtlasInfrastructure != "" {
		atlas = fmt.Sprintf("(Infrastructure: '%s' Join: %v)", config.AtlasInfrastructure, config.AtlasJoin)
	}

	// Let the agent know we've finished registration
	c.agent.StartSync()

	c.Ui.Output("Consul agent running!")
	c.Ui.Info(fmt.Sprintf("       Version: '%s'", c.HumanVersion))
	c.Ui.Info(fmt.Sprintf("     Node name: '%s'", config.NodeName))
	c.Ui.Info(fmt.Sprintf("    Datacenter: '%s'", config.Datacenter))
	c.Ui.Info(fmt.Sprintf("        Server: %v (bootstrap: %v)", config.Server, config.Bootstrap))
	c.Ui.Info(fmt.Sprintf("   Client Addr: %v (HTTP: %d, HTTPS: %d, DNS: %d, RPC: %d)", config.ClientAddr,
		config.Ports.HTTP, config.Ports.HTTPS, config.Ports.DNS, config.Ports.RPC))
	c.Ui.Info(fmt.Sprintf("  Cluster Addr: %v (LAN: %d, WAN: %d)", config.AdvertiseAddr,
		config.Ports.SerfLan, config.Ports.SerfWan))
	c.Ui.Info(fmt.Sprintf("Gossip encrypt: %v, RPC-TLS: %v, TLS-Incoming: %v",
		gossipEncrypted, config.VerifyOutgoing, config.VerifyIncoming))
	c.Ui.Info(fmt.Sprintf("         Atlas: %s", atlas))

	// Enable log streaming
	c.Ui.Info("")
	c.Ui.Output("Log data will now stream in as it occurs:\n")
	logGate.Flush()

	// Start retry join process
	errCh := make(chan struct{})
	go c.retryJoin(config, errCh)

	// Start retry -wan join process
	errWanCh := make(chan struct{})
	go c.retryJoinWan(config, errWanCh)

	// Wait for exit
	return c.handleSignals(config, errCh, errWanCh)
}

// handleSignals blocks until we get an exit-causing signal
func (c *Command) handleSignals(config *Config, retryJoin <-chan struct{}, retryJoinWan <-chan struct{}) int {
	signalCh := make(chan os.Signal, 4)
	signal.Notify(signalCh, os.Interrupt, syscall.SIGTERM, syscall.SIGHUP)

	// Wait for a signal
WAIT:
	var sig os.Signal
	select {
	case s := <-signalCh:
		sig = s
	case <-c.rpcServer.ReloadCh():
		sig = syscall.SIGHUP
	case <-c.ShutdownCh:
		sig = os.Interrupt
	case <-retryJoin:
		return 1
	case <-retryJoinWan:
		return 1
	case <-c.agent.ShutdownCh():
		// Agent is already shutdown!
		return 0
	}
	c.Ui.Output(fmt.Sprintf("Caught signal: %v", sig))

	// Check if this is a SIGHUP
	if sig == syscall.SIGHUP {
		if conf := c.handleReload(config); conf != nil {
			config = conf
		}
		goto WAIT
	}

	// Check if we should do a graceful leave
	graceful := false
	if sig == os.Interrupt && !(*config.SkipLeaveOnInt) {
		graceful = true
	} else if sig == syscall.SIGTERM && (*config.LeaveOnTerm) {
		graceful = true
	}

	// Bail fast if not doing a graceful leave
	if !graceful {
		return 1
	}

	// Attempt a graceful leave
	gracefulCh := make(chan struct{})
	c.Ui.Output("Gracefully shutting down agent...")
	go func() {
		if err := c.agent.Leave(); err != nil {
			c.Ui.Error(fmt.Sprintf("Error: %s", err))
			return
		}
		close(gracefulCh)
	}()

	// Wait for leave or another signal
	select {
	case <-signalCh:
		return 1
	case <-time.After(gracefulTimeout):
		return 1
	case <-gracefulCh:
		return 0
	}
}

// handleReload is invoked when we should reload our configs, e.g. SIGHUP
func (c *Command) handleReload(config *Config) *Config {
	c.Ui.Output("Reloading configuration...")
	newConf := c.readConfig()
	if newConf == nil {
		c.Ui.Error(fmt.Sprintf("Failed to reload configs"))
		return config
	}

	// Change the log level
	minLevel := logutils.LogLevel(strings.ToUpper(newConf.LogLevel))
	if ValidateLevelFilter(minLevel, c.logFilter) {
		c.logFilter.SetMinLevel(minLevel)
	} else {
		c.Ui.Error(fmt.Sprintf(
			"Invalid log level: %s. Valid log levels are: %v",
			minLevel, c.logFilter.Levels))

		// Keep the current log level
		newConf.LogLevel = config.LogLevel
	}

	// Bulk update the services and checks
	c.agent.PauseSync()
	defer c.agent.ResumeSync()

	// Snapshot the current state, and restore it afterwards
	snap := c.agent.snapshotCheckState()
	defer c.agent.restoreCheckState(snap)

	// First unload all checks and services. This lets us begin the reload
	// with a clean slate.
	if err := c.agent.unloadServices(); err != nil {
		c.Ui.Error(fmt.Sprintf("Failed unloading services: %s", err))
		return nil
	}
	if err := c.agent.unloadChecks(); err != nil {
		c.Ui.Error(fmt.Sprintf("Failed unloading checks: %s", err))
		return nil
	}

	// Reload services and check definitions.
	if err := c.agent.loadServices(newConf); err != nil {
		c.Ui.Error(fmt.Sprintf("Failed reloading services: %s", err))
		return nil
	}
	if err := c.agent.loadChecks(newConf); err != nil {
		c.Ui.Error(fmt.Sprintf("Failed reloading checks: %s", err))
		return nil
	}

	// Get the new client listener addr
	httpAddr, err := newConf.ClientListener(config.Addresses.HTTP, config.Ports.HTTP)
	if err != nil {
		c.Ui.Error(fmt.Sprintf("Failed to determine HTTP address: %v", err))
	}

	// Deregister the old watches
	for _, wp := range config.WatchPlans {
		wp.Stop()
	}

	// Register the new watches
	for _, wp := range newConf.WatchPlans {
		go func(wp *watch.WatchPlan) {
			wp.Handler = makeWatchHandler(c.logOutput, wp.Exempt["handler"], &c.agent.reapLock)
			wp.LogOutput = c.logOutput
			if err := wp.Run(httpAddr.String()); err != nil {
				c.Ui.Error(fmt.Sprintf("Error running watch: %v", err))
			}
		}(wp)
	}

	// Reload SCADA client if we have a change
	if newConf.AtlasInfrastructure != config.AtlasInfrastructure ||
		newConf.AtlasToken != config.AtlasToken ||
		newConf.AtlasEndpoint != config.AtlasEndpoint {
		if err := c.setupScadaConn(newConf); err != nil {
			c.Ui.Error(fmt.Sprintf("Failed reloading SCADA client: %s", err))
			return nil
		}
	}

	return newConf
}

// startScadaClient is used to start a new SCADA provider and listener,
// replacing any existing listeners.
func (c *Command) setupScadaConn(config *Config) error {
	// Shut down existing SCADA listeners
	if c.scadaProvider != nil {
		c.scadaProvider.Shutdown()
	}
	if c.scadaHttp != nil {
		c.scadaHttp.Shutdown()
	}

	// No-op if we don't have an infrastructure
	if config.AtlasInfrastructure == "" {
		return nil
	}

	scadaConfig := &scada.Config{
		Service:      "consul",
		Version:      fmt.Sprintf("%s%s", config.Version, config.VersionPrerelease),
		ResourceType: "infrastructures",
		Meta: map[string]string{
			"auto-join":  strconv.FormatBool(config.AtlasJoin),
			"datacenter": config.Datacenter,
			"server":     strconv.FormatBool(config.Server),
		},
		Atlas: scada.AtlasConfig{
			Endpoint:       config.AtlasEndpoint,
			Infrastructure: config.AtlasInfrastructure,
			Token:          config.AtlasToken,
		},
	}

	// Create the new provider and listener
	c.Ui.Output("Connecting to Atlas: " + config.AtlasInfrastructure)
	provider, list, err := scada.NewHTTPProvider(scadaConfig, c.logOutput)
	if err != nil {
		return err
	}
	c.scadaProvider = provider
	c.scadaHttp = newScadaHttp(c.agent, list)
	return nil
}

func (c *Command) Synopsis() string {
	return "Runs a Consul agent"
}

func (c *Command) Help() string {
	helpText := `
Usage: consul agent [options]

  Starts the Consul agent and runs until an interrupt is received. The
  agent represents a single node in a cluster.

Options:

  -advertise=addr          Sets the advertise address to use
  -advertise-wan=addr      Sets address to advertise on wan instead of advertise addr
  -atlas=org/name          Sets the Atlas infrastructure name, enables SCADA.
  -atlas-join              Enables auto-joining the Atlas cluster
  -atlas-token=token       Provides the Atlas API token
  -atlas-endpoint=1.2.3.4  The address of the endpoint for Atlas integration.
  -bootstrap               Sets server to bootstrap mode
  -bind=0.0.0.0            Sets the bind address for cluster communication
  -http-port=8500          Sets the HTTP API port to listen on
  -bootstrap-expect=0      Sets server to expect bootstrap mode.
  -client=127.0.0.1        Sets the address to bind for client access.
                           This includes RPC, DNS, HTTP and HTTPS (if configured)
  -config-file=foo         Path to a JSON file to read configuration from.
                           This can be specified multiple times.
  -config-dir=foo          Path to a directory to read configuration files
                           from. This will read every file ending in ".json"
                           as configuration in this directory in alphabetical
                           order. This can be specified multiple times.
  -data-dir=path           Path to a data directory to store agent state
  -dev                     Starts the agent in development mode.
  -recursor=1.2.3.4        Address of an upstream DNS server.
                           Can be specified multiple times.
  -dc=east-aws             Datacenter of the agent (deprecated: use 'datacenter' instead).
  -datacenter=east-aws     Datacenter of the agent.
  -encrypt=key             Provides the gossip encryption key
  -join=1.2.3.4            Address of an agent to join at start time.
                           Can be specified multiple times.
  -join-wan=1.2.3.4        Address of an agent to join -wan at start time.
                           Can be specified multiple times.
  -retry-join=1.2.3.4      Address of an agent to join at start time with
                           retries enabled. Can be specified multiple times.
  -retry-interval=30s      Time to wait between join attempts.
  -retry-max=0             Maximum number of join attempts. Defaults to 0, which
                           will retry indefinitely.
  -retry-join-wan=1.2.3.4  Address of an agent to join -wan at start time with
                           retries enabled. Can be specified multiple times.
  -retry-interval-wan=30s  Time to wait between join -wan attempts.
  -retry-max-wan=0         Maximum number of join -wan attempts. Defaults to 0, which
                           will retry indefinitely.
  -log-level=info          Log level of the agent.
  -node=hostname           Name of this node. Must be unique in the cluster
  -protocol=N              Sets the protocol version. Defaults to latest.
  -rejoin                  Ignores a previous leave and attempts to rejoin the cluster.
  -server                  Switches agent to server mode.
  -syslog                  Enables logging to syslog
  -ui                      Enables the built-in static web UI server
  -ui-dir=path             Path to directory containing the Web UI resources
  -pid-file=path           Path to file to store agent PID

 `
	return strings.TrimSpace(helpText)
}
