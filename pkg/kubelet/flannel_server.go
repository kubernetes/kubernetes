package kubelet

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"

	client "k8s.io/kubernetes/pkg/client/unversioned"

	flannelSubnet "github.com/coreos/flannel/subnet"
	"github.com/golang/glog"
	"github.com/gorilla/mux"
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
		glog.Error("Error JSON encoding response: %v", err)
	}
}

// GET /{network}/config
func handleGetNetworkConfig(rw http.ResponseWriter, req *http.Request) {
	defer req.Body.Close()

	checkNetwork(req)
	b, err := ioutil.ReadFile("flannel-config.json")
	if err != nil {
		glog.Fatalf("%v", err)
	}
	glog.Infof("Network config %+v", string(b))

	cfg, err := flannelSubnet.ParseConfig(string(b))
	if err != nil {
		glog.Fatalf("%v", err)
	}
	jsonResponse(rw, http.StatusOK, cfg)
}

func handleAcquireLease(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	checkNetwork(r)
	attrs := flannelSubnet.LeaseAttrs{}
	if err := json.NewDecoder(r.Body).Decode(&attrs); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, "JSON decoding error: ", err)
		return
	}
	glog.Infof("Requested lease attributes public ip %+v backend %+v backend data %+v", attrs.PublicIP, attrs.BackendType, string(attrs.BackendData))
}

func handle(rw http.ResponseWriter, req *http.Request) {
	defer req.Body.Close()

	checkNetwork(req)
}

func bindHandler(str string, h handler) http.HandlerFunc {
	return func(resp http.ResponseWriter, req *http.Request) {
		glog.Infof(str)
		glog.Infof("%+v", req.URL)
		h(resp, req)
	}
}

type FlannelServer struct {
	kubeClient client.Interface
}

func (f *FlannelServer) Run() {
	r := mux.NewRouter()
	r.HandleFunc("/v1/{network}/config", bindHandler("config GET", handleGetNetworkConfig)).Methods("GET")
	r.HandleFunc("/v1/{network}/leases", bindHandler("leases POST", handleAcquireLease)).Methods("POST")
	r.HandleFunc("/v1/{network}/leases/{subnet}", bindHandler("leases/subnet PUT", handle)).Methods("PUT")
	r.HandleFunc("/v1/{network}/leases", bindHandler("leases GET", handle)).Methods("GET")
	r.HandleFunc("/v1/", bindHandler("v1 GET", handle)).Methods("GET")

	glog.V(1).Infof("serving on :8081")
	http.ListenAndServe(":8081", r)
}
