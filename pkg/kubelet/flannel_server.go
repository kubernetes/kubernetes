package kubelet

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"time"

	//"k8s.io/kubernetes/pkg/api"
	//"k8s.io/kubernetes/pkg/client/unversioned/cache"
	//"k8s.io/kubernetes/pkg/controller/framework"
	//"k8s.io/kubernetes/pkg/fields"
	//"k8s.io/kubernetes/pkg/runtime"
	//"k8s.io/kubernetes/pkg/watch"

	client "k8s.io/kubernetes/pkg/client/unversioned"

	flannelSubnet "github.com/coreos/flannel/subnet"
	"github.com/golang/glog"
	"github.com/gorilla/mux"
)

const (
	flannelBackendDataAnnotation = "kubernetes.io/flannel.BackendData"
	day                          = 24 * time.Hour
	networkType                  = "vxlan"
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
	glog.Infof("Retrieved resource version %v", rv)

	wr, err := f.WatchLeases(rv)
	if err != nil {
		badResponse(w, err)
		return
	}
	glog.Infof("Received watch response %+v", wr)
	jsonResponse(w, http.StatusOK, wr)
}

func (f *FlannelServer) handle(rw http.ResponseWriter, req *http.Request) {
	defer req.Body.Close()

	checkNetwork(req)
}

type FlannelServer struct {
	*KubeFlannelClient
}

func NewFlannelServer(client *client.Client) *FlannelServer {
	return &FlannelServer{NewKubeFlannelClient(client)}
}

func (f *FlannelServer) Run(stopCh <-chan struct{}) {
	r := mux.NewRouter()
	r.HandleFunc("/v1/{network}/config", bindHandler("config GET", f.handleGetNetworkConfig)).Methods("GET")
	r.HandleFunc("/v1/{network}/leases", bindHandler("leases POST", f.handleAcquireLease)).Methods("POST")
	r.HandleFunc("/v1/{network}/leases/{subnet}", bindHandler("leases/subnet PUT", f.handle)).Methods("PUT")
	r.HandleFunc("/v1/{network}/leases", bindHandler("leases GET", f.handleWatch)).Methods("GET")
	r.HandleFunc("/v1/", bindHandler("v1 GET", f.handle)).Methods("GET")

	// TODO: Clean shutdown server on stop signal
	glog.V(1).Infof("serving on :8081")
	http.ListenAndServe(":8081", r)
}
