package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/BurntSushi/toml"
	"github.com/codegangsta/cli"
	"github.com/docker/docker/opts"
	"github.com/docker/docker/pkg/discovery"
	"github.com/docker/docker/pkg/reexec"

	"github.com/docker/docker/api/types/network"
	"github.com/docker/docker/pkg/term"
	"github.com/docker/libnetwork"
	"github.com/docker/libnetwork/api"
	"github.com/docker/libnetwork/cluster"
	"github.com/docker/libnetwork/config"
	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/netutils"
	"github.com/docker/libnetwork/options"
	"github.com/docker/libnetwork/types"
	"github.com/gorilla/mux"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

const (
	// DefaultHTTPHost is used if only port is provided to -H flag e.g. docker -d -H tcp://:8080
	DefaultHTTPHost = "0.0.0.0"
	// DefaultHTTPPort is the default http port used by dnet
	DefaultHTTPPort = 2385
	// DefaultUnixSocket exported
	DefaultUnixSocket = "/var/run/dnet.sock"
	cfgFileEnv        = "LIBNETWORK_CFG"
	defaultCfgFile    = "/etc/default/libnetwork.toml"
	defaultHeartbeat  = time.Duration(10) * time.Second
	ttlFactor         = 2
)

var epConn *dnetConnection

func main() {
	if reexec.Init() {
		return
	}

	_, stdout, stderr := term.StdStreams()
	logrus.SetOutput(stderr)

	err := dnetApp(stdout, stderr)
	if err != nil {
		os.Exit(1)
	}
}

// ParseConfig parses the libnetwork configuration file
func (d *dnetConnection) parseOrchestrationConfig(tomlCfgFile string) error {
	dummy := &dnetConnection{}

	if _, err := toml.DecodeFile(tomlCfgFile, dummy); err != nil {
		return err
	}

	if dummy.Orchestration != nil {
		d.Orchestration = dummy.Orchestration
	}
	return nil
}

func (d *dnetConnection) parseConfig(cfgFile string) (*config.Config, error) {
	if strings.Trim(cfgFile, " ") == "" {
		cfgFile = os.Getenv(cfgFileEnv)
		if strings.Trim(cfgFile, " ") == "" {
			cfgFile = defaultCfgFile
		}
	}

	if err := d.parseOrchestrationConfig(cfgFile); err != nil {
		return nil, err
	}
	return config.ParseConfig(cfgFile)
}

func processConfig(cfg *config.Config) []config.Option {
	options := []config.Option{}
	if cfg == nil {
		return options
	}

	dn := "bridge"
	if strings.TrimSpace(cfg.Daemon.DefaultNetwork) != "" {
		dn = cfg.Daemon.DefaultNetwork
	}
	options = append(options, config.OptionDefaultNetwork(dn))

	dd := "bridge"
	if strings.TrimSpace(cfg.Daemon.DefaultDriver) != "" {
		dd = cfg.Daemon.DefaultDriver
	}
	options = append(options, config.OptionDefaultDriver(dd))

	if cfg.Daemon.Labels != nil {
		options = append(options, config.OptionLabels(cfg.Daemon.Labels))
	}

	if dcfg, ok := cfg.Scopes[datastore.GlobalScope]; ok && dcfg.IsValid() {
		options = append(options, config.OptionKVProvider(dcfg.Client.Provider))
		options = append(options, config.OptionKVProviderURL(dcfg.Client.Address))
	}

	dOptions, err := startDiscovery(&cfg.Cluster)
	if err != nil {
		logrus.Infof("Skipping discovery : %s", err.Error())
	} else {
		options = append(options, dOptions...)
	}

	return options
}

func startDiscovery(cfg *config.ClusterCfg) ([]config.Option, error) {
	if cfg == nil {
		return nil, errors.New("discovery requires a valid configuration")
	}

	hb := time.Duration(cfg.Heartbeat) * time.Second
	if hb == 0 {
		hb = defaultHeartbeat
	}
	logrus.Infof("discovery : %s %s", cfg.Discovery, hb.String())
	d, err := discovery.New(cfg.Discovery, hb, ttlFactor*hb, map[string]string{})
	if err != nil {
		return nil, err
	}

	if cfg.Address == "" {
		iface, err := net.InterfaceByName("eth0")
		if err != nil {
			return nil, err
		}
		addrs, err := iface.Addrs()
		if err != nil || len(addrs) == 0 {
			return nil, err
		}
		ip, _, _ := net.ParseCIDR(addrs[0].String())
		cfg.Address = ip.String()
	}

	if ip := net.ParseIP(cfg.Address); ip == nil {
		return nil, errors.New("address config should be either ipv4 or ipv6 address")
	}

	if err := d.Register(cfg.Address + ":0"); err != nil {
		return nil, err
	}

	options := []config.Option{config.OptionDiscoveryWatcher(d), config.OptionDiscoveryAddress(cfg.Address)}
	go func() {
		for {
			select {
			case <-time.After(hb):
				if err := d.Register(cfg.Address + ":0"); err != nil {
					logrus.Warn(err)
				}
			}
		}
	}()
	return options, nil
}

func dnetApp(stdout, stderr io.Writer) error {
	app := cli.NewApp()

	app.Name = "dnet"
	app.Usage = "A self-sufficient runtime for container networking."
	app.Flags = dnetFlags
	app.Before = processFlags
	app.Commands = dnetCommands

	app.Run(os.Args)
	return nil
}

func createDefaultNetwork(c libnetwork.NetworkController) {
	nw := c.Config().Daemon.DefaultNetwork
	d := c.Config().Daemon.DefaultDriver
	createOptions := []libnetwork.NetworkOption{}
	genericOption := options.Generic{}

	if nw != "" && d != "" {
		// Bridge driver is special due to legacy reasons
		if d == "bridge" {
			genericOption[netlabel.GenericData] = map[string]string{
				"BridgeName":    "docker0",
				"DefaultBridge": "true",
			}
			createOptions = append(createOptions,
				libnetwork.NetworkOptionGeneric(genericOption),
				ipamOption(nw))
		}

		if n, err := c.NetworkByName(nw); err == nil {
			logrus.Debugf("Default network %s already present. Deleting it", nw)
			if err = n.Delete(); err != nil {
				logrus.Debugf("Network could not be deleted: %v", err)
				return
			}
		}

		_, err := c.NewNetwork(d, nw, "", createOptions...)
		if err != nil {
			logrus.Errorf("Error creating default network : %s : %v", nw, err)
		}
	}
}

type dnetConnection struct {
	// proto holds the client protocol i.e. unix.
	proto string
	// addr holds the client address.
	addr          string
	Orchestration *NetworkOrchestration
	configEvent   chan cluster.ConfigEventType
}

// NetworkOrchestration exported
type NetworkOrchestration struct {
	Agent   bool
	Manager bool
	Bind    string
	Peer    string
}

func (d *dnetConnection) dnetDaemon(cfgFile string) error {
	if err := startTestDriver(); err != nil {
		return fmt.Errorf("failed to start test driver: %v", err)
	}

	cfg, err := d.parseConfig(cfgFile)
	var cOptions []config.Option
	if err == nil {
		cOptions = processConfig(cfg)
	} else {
		logrus.Errorf("Error parsing config %v", err)
	}

	bridgeConfig := options.Generic{
		"EnableIPForwarding": true,
		"EnableIPTables":     true,
	}

	bridgeOption := options.Generic{netlabel.GenericData: bridgeConfig}

	cOptions = append(cOptions, config.OptionDriverConfig("bridge", bridgeOption))

	controller, err := libnetwork.New(cOptions...)
	if err != nil {
		fmt.Println("Error starting dnetDaemon :", err)
		return err
	}
	controller.SetClusterProvider(d)

	if d.Orchestration.Agent || d.Orchestration.Manager {
		d.configEvent <- cluster.EventNodeReady
	}

	createDefaultNetwork(controller)
	httpHandler := api.NewHTTPHandler(controller)
	r := mux.NewRouter().StrictSlash(false)
	post := r.PathPrefix("/{.*}/networks").Subrouter()
	post.Methods("GET", "PUT", "POST", "DELETE").HandlerFunc(httpHandler)
	post = r.PathPrefix("/networks").Subrouter()
	post.Methods("GET", "PUT", "POST", "DELETE").HandlerFunc(httpHandler)
	post = r.PathPrefix("/{.*}/services").Subrouter()
	post.Methods("GET", "PUT", "POST", "DELETE").HandlerFunc(httpHandler)
	post = r.PathPrefix("/services").Subrouter()
	post.Methods("GET", "PUT", "POST", "DELETE").HandlerFunc(httpHandler)
	post = r.PathPrefix("/{.*}/sandboxes").Subrouter()
	post.Methods("GET", "PUT", "POST", "DELETE").HandlerFunc(httpHandler)
	post = r.PathPrefix("/sandboxes").Subrouter()
	post.Methods("GET", "PUT", "POST", "DELETE").HandlerFunc(httpHandler)

	handleSignals(controller)
	setupDumpStackTrap()

	return http.ListenAndServe(d.addr, r)
}

func (d *dnetConnection) IsManager() bool {
	return d.Orchestration.Manager
}

func (d *dnetConnection) IsAgent() bool {
	return d.Orchestration.Agent
}

func (d *dnetConnection) GetAdvertiseAddress() string {
	return d.Orchestration.Bind
}

func (d *dnetConnection) GetDataPathAddress() string {
	return d.Orchestration.Bind
}

func (d *dnetConnection) GetLocalAddress() string {
	return d.Orchestration.Bind
}

func (d *dnetConnection) GetListenAddress() string {
	return d.Orchestration.Bind
}

func (d *dnetConnection) GetRemoteAddressList() []string {
	return []string{d.Orchestration.Peer}
}

func (d *dnetConnection) ListenClusterEvents() <-chan cluster.ConfigEventType {
	return d.configEvent
}

func (d *dnetConnection) AttachNetwork(string, string, []string) (*network.NetworkingConfig, error) {
	return nil, nil
}

func (d *dnetConnection) DetachNetwork(string, string) error {
	return nil
}

func (d *dnetConnection) UpdateAttachment(string, string, *network.NetworkingConfig) error {
	return nil
}

func (d *dnetConnection) WaitForDetachment(context.Context, string, string, string, string) error {
	return nil
}

func handleSignals(controller libnetwork.NetworkController) {
	c := make(chan os.Signal, 1)
	signals := []os.Signal{os.Interrupt, syscall.SIGTERM, syscall.SIGQUIT}
	signal.Notify(c, signals...)
	go func() {
		for range c {
			controller.Stop()
			os.Exit(0)
		}
	}()
}

func startTestDriver() error {
	mux := http.NewServeMux()
	server := httptest.NewServer(mux)
	if server == nil {
		return errors.New("Failed to start an HTTP Server")
	}

	mux.HandleFunc("/Plugin.Activate", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprintf(w, `{"Implements": ["%s"]}`, driverapi.NetworkPluginEndpointType)
	})

	mux.HandleFunc(fmt.Sprintf("/%s.GetCapabilities", driverapi.NetworkPluginEndpointType), func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprint(w, `{"Scope":"global"}`)
	})

	mux.HandleFunc(fmt.Sprintf("/%s.CreateNetwork", driverapi.NetworkPluginEndpointType), func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprint(w, "null")
	})

	mux.HandleFunc(fmt.Sprintf("/%s.DeleteNetwork", driverapi.NetworkPluginEndpointType), func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprint(w, "null")
	})

	mux.HandleFunc(fmt.Sprintf("/%s.CreateEndpoint", driverapi.NetworkPluginEndpointType), func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprint(w, "null")
	})

	mux.HandleFunc(fmt.Sprintf("/%s.DeleteEndpoint", driverapi.NetworkPluginEndpointType), func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprint(w, "null")
	})

	mux.HandleFunc(fmt.Sprintf("/%s.Join", driverapi.NetworkPluginEndpointType), func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprint(w, "null")
	})

	mux.HandleFunc(fmt.Sprintf("/%s.Leave", driverapi.NetworkPluginEndpointType), func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprint(w, "null")
	})

	if err := os.MkdirAll("/etc/docker/plugins", 0755); err != nil {
		return err
	}

	if err := ioutil.WriteFile("/etc/docker/plugins/test.spec", []byte(server.URL), 0644); err != nil {
		return err
	}

	return nil
}

func newDnetConnection(val string) (*dnetConnection, error) {
	url, err := opts.ParseHost(false, val)
	if err != nil {
		return nil, err
	}
	protoAddrParts := strings.SplitN(url, "://", 2)
	if len(protoAddrParts) != 2 {
		return nil, errors.New("bad format, expected tcp://ADDR")
	}
	if strings.ToLower(protoAddrParts[0]) != "tcp" {
		return nil, errors.New("dnet currently only supports tcp transport")
	}

	return &dnetConnection{protoAddrParts[0], protoAddrParts[1], &NetworkOrchestration{}, make(chan cluster.ConfigEventType, 10)}, nil
}

func (d *dnetConnection) httpCall(method, path string, data interface{}, headers map[string][]string) (io.ReadCloser, http.Header, int, error) {
	var in io.Reader
	in, err := encodeData(data)
	if err != nil {
		return nil, nil, -1, err
	}

	req, err := http.NewRequest(method, fmt.Sprintf("%s", path), in)
	if err != nil {
		return nil, nil, -1, err
	}

	setupRequestHeaders(method, data, req, headers)

	req.URL.Host = d.addr
	req.URL.Scheme = "http"

	httpClient := &http.Client{}
	resp, err := httpClient.Do(req)
	statusCode := -1
	if resp != nil {
		statusCode = resp.StatusCode
	}
	if err != nil {
		return nil, nil, statusCode, fmt.Errorf("error when trying to connect: %v", err)
	}

	if statusCode < 200 || statusCode >= 400 {
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return nil, nil, statusCode, err
		}
		return nil, nil, statusCode, fmt.Errorf("error : %s", bytes.TrimSpace(body))
	}

	return resp.Body, resp.Header, statusCode, nil
}

func setupRequestHeaders(method string, data interface{}, req *http.Request, headers map[string][]string) {
	if data != nil {
		if headers == nil {
			headers = make(map[string][]string)
		}
		headers["Content-Type"] = []string{"application/json"}
	}

	expectedPayload := (method == "POST" || method == "PUT")

	if expectedPayload && req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", "text/plain")
	}

	if headers != nil {
		for k, v := range headers {
			req.Header[k] = v
		}
	}
}

func encodeData(data interface{}) (*bytes.Buffer, error) {
	params := bytes.NewBuffer(nil)
	if data != nil {
		if err := json.NewEncoder(params).Encode(data); err != nil {
			return nil, err
		}
	}
	return params, nil
}

func ipamOption(bridgeName string) libnetwork.NetworkOption {
	if nws, _, err := netutils.ElectInterfaceAddresses(bridgeName); err == nil {
		ipamV4Conf := &libnetwork.IpamConf{PreferredPool: nws[0].String()}
		hip, _ := types.GetHostPartIP(nws[0].IP, nws[0].Mask)
		if hip.IsGlobalUnicast() {
			ipamV4Conf.Gateway = nws[0].IP.String()
		}
		return libnetwork.NetworkOptionIpam("default", "", []*libnetwork.IpamConf{ipamV4Conf}, nil, nil)
	}
	return nil
}
