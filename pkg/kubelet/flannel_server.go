package kubelet

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	client "k8s.io/kubernetes/pkg/client/unversioned"

	flannelSubnet "github.com/coreos/flannel/subnet"
	"github.com/golang/glog"
	"github.com/gorilla/mux"
)

const (
	flannelBackendDataAnnotation = "kubernetes.io/flannel.BackendData"
	day                          = 24 * time.Hour
	flannelPort                  = 8081
	networkType                  = "vxlan"
	dockerOptsFile               = "/etc/default/docker"
	flannelSubnetKey             = "FLANNEL_SUBNET"
	flannelMtuKey                = "FLANNEL_MTU"
	dockerOptsKey                = "DOCKER_OPS"
	flannelSubnetFile            = "/var/run/flannel/subnet.env"
)

type handler func(http.ResponseWriter, *http.Request)

func checkNetwork(req *http.Request) {
	network := mux.Vars(req)["network"]
	if network != "_" {
		glog.Fatalf("Alien network %v", network)
	}
}

func jsonResponse(w http.ResponseWriter, code int, v interface{}) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		badResponse(w, err)
		return
	}
}

func badResponse(w http.ResponseWriter, err error) {
	errMsg := fmt.Sprintf("Flannel server error: %v", err)
	glog.Errorf(errMsg)
	// TODO: Take status as arg
	w.WriteHeader(http.StatusInternalServerError)
	fmt.Fprintf(w, errMsg)
}

func bindHandler(str string, h handler) http.HandlerFunc {
	return func(resp http.ResponseWriter, req *http.Request) {
		glog.Infof(str)
		glog.Infof("%+v", req.URL)
		h(resp, req)
	}
}

func getResourceVersion(u *url.URL) string {
	vals, ok := u.Query()["next"]
	if !ok {
		return ""
	}
	return vals[0]
}

// GET /{network}/config
func (f *FlannelServer) handleGetNetworkConfig(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	checkNetwork(r)
	b, err := ioutil.ReadFile("flannel-config.json")
	if err != nil {
		badResponse(w, err)
		return
	}
	glog.Infof("Network config %+v", string(b))

	cfg, err := flannelSubnet.ParseConfig(string(b))
	if err != nil {
		badResponse(w, err)
		return
	}
	jsonResponse(w, http.StatusOK, cfg)
}

// POST /{network}/leases
func (f *FlannelServer) handleAcquireLease(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	checkNetwork(r)
	attrs := &flannelSubnet.LeaseAttrs{}
	if err := json.NewDecoder(r.Body).Decode(attrs); err != nil {
		badResponse(w, err)
		return
	}

	glog.Infof("Requested lease attributes public ip %+v backend %+v backend data %+v",
		attrs.PublicIP, attrs.BackendType, string(attrs.BackendData))

	node, err := f.client.getNodeByIp(attrs.PublicIP.String())
	if err != nil {
		badResponse(w, err)
		return
	}
	if node == nil {
		w.WriteHeader(http.StatusPreconditionFailed)
		fmt.Fprint(w, "Waiting for kubelet to create node")
		return
	}
	glog.Infof("Retrieved node %v for ip %v", node.Name, attrs.PublicIP)
	subnet, err := f.client.getSubnet(node)
	if err != nil {
		badResponse(w, err)
		return
	}
	glog.Infof("Node subnet %+v", subnet)
	if err := f.client.setAnnotation(
		flannelBackendDataAnnotation,
		string(attrs.BackendData),
		node); err != nil {

		badResponse(w, err)
		return
	}
	// TODO: What's the value for no expire ttl?
	lease := &flannelSubnet.Lease{
		Subnet:     *subnet,
		Attrs:      attrs,
		Expiration: time.Now().Add(24 * time.Hour),
	}
	jsonResponse(w, http.StatusOK, lease)
}

// GET /{network}/leases?next=cursor
func (f *FlannelServer) handleWatch(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	checkNetwork(r)

	rv := getResourceVersion(r.URL)
	glog.Infof("(flannel) Retrieved resource version %v", rv)

	wr, err := f.WatchLeases(rv)
	if err != nil {
		badResponse(w, err)
		return
	}
	glog.Infof("(flannel) Received watch response %+v", wr)
	jsonResponse(w, http.StatusOK, wr)
}

func (f *FlannelServer) handle(rw http.ResponseWriter, req *http.Request) {
	defer req.Body.Close()

	checkNetwork(req)
}

type FlannelServer struct {
	*KubeFlannelClient
	port       int
	subnetFile string
}

func NewFlannelServer(client *client.Client) *FlannelServer {
	return &FlannelServer{NewKubeFlannelClient(client), flannelPort, flannelSubnetFile}
}

func (f *FlannelServer) RunServer(stopCh <-chan struct{}) {
	glog.Infof("(flannel) Starting flannel server")
	r := mux.NewRouter()
	r.HandleFunc("/v1/{network}/config", bindHandler("config GET", f.handleGetNetworkConfig)).Methods("GET")
	r.HandleFunc("/v1/{network}/leases", bindHandler("leases POST", f.handleAcquireLease)).Methods("POST")
	r.HandleFunc("/v1/{network}/leases/{subnet}", bindHandler("leases/subnet PUT", f.handle)).Methods("PUT")
	r.HandleFunc("/v1/{network}/leases", bindHandler("leases GET", f.handleWatch)).Methods("GET")
	r.HandleFunc("/v1/", bindHandler("v1 GET", f.handle)).Methods("GET")

	// TODO: Clean shutdown server on stop signal
	glog.V(1).Infof("serving on :8081")
	http.ListenAndServe(fmt.Sprintf(":%v", f.port), r)
}

func (f *FlannelServer) Handshake() (podCIDR string, err error) {
	// Flannel daemon will hang till the server comes up, kubelet will hang until
	// flannel daemon has written subnet env variables. This is the kubelet handshake.
	// To improve performance, we could defer just the configuration of the container
	// bridge till after subnet.env is written. Keeping it local is clearer for now.
	// TODO: Using a file to communicate is brittle
	if _, err = os.Stat(f.subnetFile); err != nil {
		return "", fmt.Errorf("Waiting for subnet file %v", f.subnetFile)
	}
	glog.Infof("(flannel) Found flannel subnet file %v", f.subnetFile)

	// TODO: Rest of this function is a hack.
	config, err := parseKVConfig(f.subnetFile)
	if err != nil {
		return "", err
	}
	if err = writeDockerOptsFromFlannelConfig(config); err != nil {
		return "", err
	}
	podCIDR, ok := config[flannelSubnetKey]
	if !ok {
		return "", fmt.Errorf("No flannel subnet, config %+v", config)
	}
	return podCIDR, nil
}

// Take env variables from flannel subnet env and write to /etc/docker/defaults.
func writeDockerOptsFromFlannelConfig(flannelConfig map[string]string) error {
	// TODO: Write dockeropts to unit file on systemd machines
	// https://github.com/docker/docker/issues/9889
	mtu, ok := flannelConfig[flannelMtuKey]
	if !ok {
		return fmt.Errorf("No flannel mtu, flannel config %+v", flannelConfig)
	}
	dockerOpts, err := parseKVConfig(dockerOptsFile)
	if err != nil {
		return err
	}
	opts, ok := dockerOpts[dockerOptsKey]
	if !ok {
		glog.Errorf("(flannel) Did not find docker opts, writing them")
		opts = fmt.Sprintf(
			" --bridge=cbr0 --iptables=false --ip-masq=false")
	}
	dockerOpts[dockerOptsKey] = fmt.Sprintf("%v --mtu=%v", opts, mtu)
	if err = writeKVConfig(dockerOptsFile, dockerOpts); err != nil {
		return err
	}
	return nil
}

// parseKVConfig takes a file with key-value env variables and returns a dictionary mapping the same.
func parseKVConfig(filename string) (map[string]string, error) {
	config := map[string]string{}
	if _, err := os.Stat(filename); err != nil {
		return config, err
	}
	buff, err := ioutil.ReadFile(filename)
	if err != nil {
		return config, err
	}
	str := string(buff)
	glog.Infof("(flannel) Read kv options %+v from %v", str, filename)
	for _, line := range strings.Split(str, "\n") {
		kv := strings.Split(line, "=")
		if len(kv) != 2 {
			glog.Infof("(flannel) Ignoring non key-value pair %v", kv)
			continue
		}
		config[string(kv[0])] = string(kv[1])
	}
	return config, nil
}

// writeKVConfig writes a kv map as env variables into the given file.
func writeKVConfig(filename string, kv map[string]string) error {
	if _, err := os.Stat(filename); err != nil {
		return err
	}
	content := ""
	for k, v := range kv {
		content += fmt.Sprintf("%v=\"%v\"\n", k, v)
	}
	glog.Infof("(flannel) Writing kv options %+v to %v", content, filename)
	return ioutil.WriteFile(filename, []byte(content), 0644)
}
