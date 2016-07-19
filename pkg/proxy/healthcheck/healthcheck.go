/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package healthcheck

import (
	"fmt"
	"log"
	"net"
	"net/http"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/healthcheckparser"
)

const (
	AddEndpointOp      = iota
	DeleteEndpointOp   = iota
	DeleteServiceOp    = iota
	HealthCheckRequest = iota
)

var serviceEndpointsMap ServiceEndpointsMap

func init() {
	serviceEndpointsMap = make(map[string]ServiceEndpointsList)
	// HACK HACK HACK - forcibly start listening on port 32023
	// TODO - stitch this into the kube-proxy init sequence
	ProxyHealthCheckFactory("", 32023)
}

type ProxyHealthCheckRequest struct {
	Operation   int
	ServiceUid  string
	Namespace   string
	ServiceName string
	Endpoints   *api.Endpoints

	Result          bool
	ResponseChannel *chan *ProxyHealthCheckRequest

	/* Opaque data sent back to client */
	rw  *http.ResponseWriter
	req *http.Request
}

type ServiceEndpointsList struct {
	endpoints map[string]*api.Endpoints
}

type ServiceEndpointsMap map[string]ServiceEndpointsList

type ProxyHealthChecker interface {
	EnqueueRequest(req *ProxyHealthCheckRequest)
}

type ProxyHC struct {
	requestChannel chan *ProxyHealthCheckRequest
	hostIp         string
	port           int16

	server http.Server
	// net.Handler interface for responses
	proxyHandler    ProxyHCHandler
	shutdownChannel chan bool
}

type ProxyHCHandler struct {
	phc *ProxyHC
}

func sendHealthCheckResponse(rw *http.ResponseWriter, statusCode int, error string) {
	(*rw).Header().Set("Content-Type", "text/plain")
	(*rw).WriteHeader(statusCode)
	fmt.Fprint(*rw, error)
}

func parseHttpRequest(response *http.ResponseWriter, req *http.Request) (*ProxyHealthCheckRequest, chan *ProxyHealthCheckRequest, error) {
	glog.Infof("Received Health Check on url %s", req.URL.String())
	// Sanity check and parse the healthcheck URL
	namespace, name, uid, err := healthcheckparser.ParseURL(req.URL.String())
	if err != nil {
		glog.Info("Parse failure - cannot respond to malformed healthcheck URL")
		return nil, nil, err
	}
	// TODO - logging
	glog.Infof("Parsed Healthcheck as service %s/%s (uid %s)", namespace, name, uid)
	responseChannel := make(chan *ProxyHealthCheckRequest)
	msg := &ProxyHealthCheckRequest{Operation: HealthCheckRequest,
		ResponseChannel: &responseChannel,
		ServiceUid:      uid,
		Namespace:       namespace,
		ServiceName:     name,
		rw:              response,
		req:             req,
		Result:          false,
		Endpoints:       nil,
	}
	return msg, responseChannel, nil
}

func (handler ProxyHCHandler) ServeHTTP(response http.ResponseWriter, req *http.Request) {
	// Grab the session guid from the URL and forward the request to the healtchecker
	msg, responseChannel, err := parseHttpRequest(&response, req)
	//TODO - logging
	glog.Infof("Received HC request for service uid %s from LB control plane", msg.ServiceUid)
	if err != nil {
		//TODO - Return error HTTP code
		sendHealthCheckResponse(&response, http.StatusBadRequest, fmt.Sprintf("Parse error: %s", err))
	}
	go handler.phc.EnqueueRequest(msg)
	<-responseChannel
}

// handleHealthCheckRequest - received a health check request - lookup and respond to HC.
func (handler ProxyHC) handleHealthCheckRequest(req *ProxyHealthCheckRequest) {
	defer func(r *ProxyHealthCheckRequest) { *(r.ResponseChannel) <- r }(req)
	service, ok := serviceEndpointsMap[req.ServiceUid]
	if !ok {
		sendHealthCheckResponse(req.rw, http.StatusNotFound, "Service Endpoint Not Found")
	} else {
		sendHealthCheckResponse(req.rw, http.StatusOK, fmt.Sprintf("%d Service Endpoints found", len(service.endpoints)))
	}
}

// handleMutationRequest - received a request to mutate the table
func (handler ProxyHC) handleMutationRequest(req *ProxyHealthCheckRequest) {
	fmt.Println("Received table mutation request")

}

func (handler ProxyHC) HandlerLoop() {
	for {
		select {
		case req := <-handler.requestChannel:
			//TODO - logging
			fmt.Println("Dequeued request in health check loop ")
			if req.Operation == HealthCheckRequest {
				handler.handleHealthCheckRequest(req)
			} else {
				handler.handleMutationRequest(req)
			}
		case <-handler.shutdownChannel:
			//TODO - logging
			fmt.Println("Received shutdown request")
			break
		}
	}
}

func (handler ProxyHC) Shutdown() {
	if handler.shutdownChannel != nil {
		handler.shutdownChannel <- true
	}
}

func ProxyHealthCheckFactory(ip string, port int16) *ProxyHC {
	phc := &ProxyHC{requestChannel: make(chan *ProxyHealthCheckRequest),
		hostIp:          ip,
		port:            port,
		shutdownChannel: make(chan bool)}

	readyChan := make(chan bool)
	go phc.StartListening(readyChan)

	response := <-readyChan
	if response != true {
		//TODO
		log.Fatalf("Failed to bind and listen on %s:%d\n", ip, port)
		return nil
	}
	return phc
}

func (h *ProxyHC) EnqueueRequest(req *ProxyHealthCheckRequest) {
	h.requestChannel <- req
}

func (h *ProxyHC) StartListening(readyChannel chan bool) {

	h.proxyHandler = ProxyHCHandler{phc: h}
	h.server = http.Server{Addr: fmt.Sprintf("%s:%d", h.hostIp, h.port), Handler: h.proxyHandler}

	ln, err := net.Listen("tcp", h.server.Addr)
	if err != nil {
		// TODO
		fmt.Printf("FAILED TO listen on address %s (%s)", h.server.Addr, err)
		readyChannel <- false
	}

	readyChannel <- true
	go h.HandlerLoop()
	defer h.Shutdown()

	err = h.server.Serve(ln)
	if err != nil {
		// TODO
		fmt.Printf("Proxy HealthCheck listen socket failure (%s)", err)
		// TODO - what do we do here ?
	}
}
