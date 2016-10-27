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

// Based on the network-tester, this implements a network monitor that can
// be used to continuously monitor connectivity between a netmonitor container
// and a set of peer netmonitor containers (discovered through namespace/service
// query).
//
// Connectivity between the containers is monitored using a lightweight web
// server (for TCP traffic).  This container tracks when the last message was
// responded to by each peer, _and_ when it last received a request from the
// peer - thus this tracks both inbound and outbound connectivity separately.
//
// The container also serves webserver UI to allow a client to receive the
// connectivity information between the pods.  It serves the following endpoints:
//
// /quit       : to shut down
// /summary    : to see a summary status of connections
// /details    : to see the detailed internal state
//
// The internal facing webserver serves up the following:
// /ping
//
// The external query UI uses a different port to the internal inter-container "ping" port.
// This allows policy lockdown of the inter-container port whilst still being able to
// monitor the connectivity through the UI port.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/sets"
)

// Information about an individual peer message.
type TCPMessage struct {
	// Values in this structure require the State lock to be held before
	// reading or writing.
	NextIndex           int
	LastSuccessIndex    int
	LastSuccessTime     *time.Time
	TimeBetweenMessages []time.Duration
}

// State tracks the internal connection state of all peers.
type State struct {
	// Hostname is set once and never changed-- it's always safe to read.
	Hostname string

	// The below fields require that lock is held before reading or writing.
	TCPOutbound map[string]TCPMessage
	TCPInbound  map[string]TCPMessage

	lock sync.Mutex
}

// HTTPPingMessage is the format that (json encoded) requests to the /ping handler should take,
// and is returned directly as the response.
type HTTPPingMessage struct {
	Index    int
	SourceID string
}

// Summary is returned by /summary
type Summary struct {
	TCPNumOutboundConnected int
	TCPNumInboundConnected  int
}

var (
	// Runtime flags
	tcpPort           = flag.Int("tcp-port", 8081, "Local TCP Port number used for pod-to-pod connectivity testing.")
	queryPort         = flag.Int("query-port", 8080, "Local TCP Port number used for HTTP queries to obtain summary and detailed connectivity information.")
	remoteTCPPortName = flag.String("remote-tcp-port-name", "net-monitor-tcp-ping", "Port name used to identify the remote pod-to-pod TCP testing ports.")
	namespace         = flag.String("namespace", "default", "Namespace containing peer network monitor pods.")
	service           = flag.String("service", "netmonitor", "Service containing peer network monitor pods.")
	expireFactor      = flag.Int64("expiration-factor", 3, "Factor to multiple time between received messages which is then used to tune the expiration time (chosen as the maximum of the current value and the factored time).")
	delayShutdown     = flag.Int("delay-shutdown", 0, "Number of seconds to delay shutdown when receiving SIGTERM.")

	// Number of message times to keep track of
	msgHistory = 3
)

// Log an error message to stdout.
func logErr(err error) {
	if err != nil {
		log.Printf("Error: %v", err)
	}
}

// Look at the current TCP message data to determine if the TCP connection is currently
// connected.
func (t *TCPMessage) isConnected(now time.Time) bool {
	if t.LastSuccessTime == nil {
		return false
	}

	// We only consider ourselves connected when we have received at least a
	// couple of consecutive messages so that we know what the approximate time
	// between messages is - without that we can't determine what constitutes
	// a fail.
	if len(t.TimeBetweenMessages) == 0 {
		return false
	}

	// Calculate the average time between messages.  We multiply this by our
	// expiration factor to determine when there is no connection.
	timeBetweenMessages := time.Duration(0)
	for i := 0; i < len(t.TimeBetweenMessages); i++ {
		timeBetweenMessages += t.TimeBetweenMessages[i]
	}
	timeBetweenMessages = timeBetweenMessages / time.Duration(len(t.TimeBetweenMessages))

	// Check if the time of last success indicates that the channel is still
	// connected.
	return now.Sub(*t.LastSuccessTime) < (timeBetweenMessages * time.Duration(*expireFactor))
}

// serveSummary returns a JSON dictionary containing a summary of
// counts of successful inbound and outbound peers connections.
// e.g.
//   {
//     "TCPNumInboundConnected": 10,
//     "TCPNumOutboundConnected": 4
//   }
func (s *State) serveSummary(w http.ResponseWriter, r *http.Request) {
	log.Printf("Serving summary")
	s.lock.Lock()
	defer s.lock.Unlock()

	now := time.Now()
	summary := Summary{
		TCPNumInboundConnected:  0,
		TCPNumOutboundConnected: 0,
	}

	for _, v := range s.TCPOutbound {
		if v.isConnected(now) {
			summary.TCPNumOutboundConnected += 1
		}
	}

	for _, v := range s.TCPInbound {
		if v.isConnected(now) {
			summary.TCPNumInboundConnected += 1
		}
	}

	w.WriteHeader(http.StatusOK)
	b, err := json.MarshalIndent(&summary, "", "\t")
	logErr(err)
	_, err = w.Write(b)
	logErr(err)
}

// serveDetails writes our json encoded state
func (s *State) serveDetails(w http.ResponseWriter, r *http.Request) {
	log.Printf("Serving details")
	s.lock.Lock()
	defer s.lock.Unlock()
	w.WriteHeader(http.StatusOK)
	b, err := json.MarshalIndent(s, "", "\t")
	logErr(err)
	_, err = w.Write(b)
	logErr(err)
}

// servePing responds to a ping from a peer, and records the peer contact in our
// received state.
func (s *State) servePing(w http.ResponseWriter, r *http.Request) {
	log.Printf("Serving ping")
	defer r.Body.Close()
	s.lock.Lock()
	defer s.lock.Unlock()
	w.WriteHeader(http.StatusOK)
	var msg HTTPPingMessage
	logErr(json.NewDecoder(r.Body).Decode(&msg))
	if msg.SourceID == "" {
		logErr(fmt.Errorf("%v: Got request with no source ID", s.Hostname))
	} else {
		now := time.Now()
		stored := s.TCPInbound[msg.SourceID]
		if msg.Index >= stored.NextIndex {
			if msg.Index == stored.NextIndex && stored.LastSuccessTime != nil {
				// This is a consecutive message so we can record the time between
				// messages.  We use this to adjust our expiration times based
				// on load.
				stored.TimeBetweenMessages = append(stored.TimeBetweenMessages, now.Sub(*stored.LastSuccessTime))
				if len(stored.TimeBetweenMessages) > msgHistory {
					stored.TimeBetweenMessages = stored.TimeBetweenMessages[1:]
				}
			}

			// Store the index and the current time.
			stored.LastSuccessTime = &now
			stored.LastSuccessIndex = msg.Index

			// Update the next index we expect.
			stored.NextIndex = msg.Index + 1
		}

		// Update the map to store the data for this connection.
		s.TCPInbound[msg.SourceID] = stored
	}

	// Send the original request back as the response.
	logErr(json.NewEncoder(w).Encode(&msg))
}

// Monitor a single TCP peer by sending an HTTP ping and waiting for the response.
func (s *State) monitorTCPPeer(endpoint string) {
	var index int

	// Obtain the current index for this peer and increment it in our shared data.  We
	// need to hold the lock for this, but only want to hold it for the minimum amount
	// of time.
	func() {
		log.Printf("Processing endpoing: %s", endpoint)
		s.lock.Lock()
		defer s.lock.Unlock()

		data := s.TCPOutbound[endpoint]
		index = data.NextIndex
		data.NextIndex += 1
		s.TCPOutbound[endpoint] = data
	}()

	// Send the HTTP ping request.
	log.Printf("Attempting to contact %s", endpoint)
	body, err := json.Marshal(&HTTPPingMessage{
		Index:    index,
		SourceID: s.Hostname,
	})
	if err != nil {
		log.Fatalf("json marshal error: %v", err)
	}
	resp, err := http.Post(endpoint+"/ping", "application/json", bytes.NewReader(body))
	if err != nil {
		log.Printf("Warning: unable to contact the endpoint %q: %v", endpoint, err)
		return
	}
	defer resp.Body.Close()

	// Read and unmarshal the response.
	body, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Printf("Warning: unable to read response from '%v': '%v'", endpoint, err)
		return
	}
	log.Printf("Response from endpoint: %v", string(body))

	var response HTTPPingMessage
	err = json.Unmarshal(body, &response)
	if err != nil {
		log.Printf("Warning: unable to unmarshal response (%v) from '%v': '%v'", string(body), endpoint, err)
		return
	}

	// Update our state information based on the response.
	func() {
		log.Printf("Successful response")
		s.lock.Lock()
		defer s.lock.Unlock()

		now := time.Now()
		data := s.TCPOutbound[endpoint]
		log.Printf("Index: %d, %d", index, data.LastSuccessIndex)
		if index == data.LastSuccessIndex+1 && data.LastSuccessTime != nil {
			// This is a consecutive message so we can record the time between
			// messages.  We use this to adjust our expiration times based
			// on load.
			log.Printf("Update time between messages")
			data.TimeBetweenMessages = append(data.TimeBetweenMessages, now.Sub(*data.LastSuccessTime))
			if len(data.TimeBetweenMessages) > msgHistory {
				data.TimeBetweenMessages = data.TimeBetweenMessages[1:]
			}
		}

		if index > data.LastSuccessIndex {
			log.Printf("Update last success time")
			data.LastSuccessIndex = index
			data.LastSuccessTime = &now
		}
		s.TCPOutbound[endpoint] = data
	}()
}

// Find all sibling pods in the service and post to their /write handler.
func (s *State) monitorPeers() {
	log.Printf("Monitor peers")
	client, err := client.NewInCluster()
	if err != nil {
		log.Fatalf("Unable to create client; error: %v\n", err)
	}
	// Double check that that worked by getting the server version.
	if v, err := client.Discovery().ServerVersion(); err != nil {
		log.Fatalf("Unable to get server version: %v\n", err)
	} else {
		log.Printf("Server version: %#v\n", v)
	}

	// Repeatedly obtain the endpoint list and monitor the TCP connections.
	for {
		tcp_eps := getEndpoints(client)
		for ep := range tcp_eps {
			s.monitorTCPPeer(ep)
		}
		time.Sleep(2 * time.Second)
	}
}

// Listen to a particular port and serve responses.
func listenAndServe(port int) {
	err := http.ListenAndServe(fmt.Sprintf("0.0.0.0:%d", port), nil)
	if err != nil {
		log.Fatal(err)
	}
}

// getEndpoints returns the endpoints as set of String:
// -  TCP endpoints:  "http://{ip}:{port}"
func getEndpoints(client *client.Client) sets.String {
	endpoints, err := client.Endpoints(*namespace).Get(*service)
	tcp_eps := sets.String{}
	if err != nil {
		log.Printf("Unable to read the endpoints for %v/%v: %v.", *namespace, *service, err)
		return tcp_eps
	}
	for _, ss := range endpoints.Subsets {
		for _, a := range ss.Addresses {
			for _, p := range ss.Ports {
				// Inter-pod ping ports are discovered by name.
				if p.Protocol == api.ProtocolTCP && p.Name == *remoteTCPPortName {
					tcp_eps.Insert(fmt.Sprintf("http://%s:%d", a.IP, p.Port))
				}
			}
		}
	}
	return tcp_eps
}

// Main.  Parse arguments, initialize state, start monitoring and start the web servers.
func main() {
	log.Printf("Parsing arguments")
	flag.Parse()

	if *service == "" {
		log.Fatal("Must provide -service flag.")
	}

	hostname, err := os.Hostname()
	if err != nil {
		log.Fatalf("Error getting hostname: %v", err)
	}

	if *delayShutdown > 0 {
		log.Printf("Configure delayed shutdown")
		termCh := make(chan os.Signal)
		signal.Notify(termCh, syscall.SIGTERM)
		go func() {
			<-termCh
			log.Printf("Sleeping %d seconds before exit ...", *delayShutdown)
			time.Sleep(time.Duration(*delayShutdown) * time.Second)
			os.Exit(0)
		}()
	}

	log.Printf("Initialize state")
	state := State{
		Hostname:    hostname,
		TCPOutbound: map[string]TCPMessage{},
		TCPInbound:  map[string]TCPMessage{},
	}

	go state.monitorPeers()

	log.Printf("Configure handler functions")
	http.HandleFunc("/quit", func(w http.ResponseWriter, r *http.Request) {
		os.Exit(0)
	})
	http.HandleFunc("/summary", state.serveSummary)
	http.HandleFunc("/details", state.serveDetails)
	http.HandleFunc("/ping", state.servePing)

	// Start up the server on the required ports - by default the inter-pod communication
	// is on a different port to the UX.  Both can be handled by the same default
	// handler though.
	log.Printf("Listening on inter-pod port: %d", *tcpPort)
	go listenAndServe(*tcpPort)
	if *queryPort != *tcpPort {
		log.Printf("Listening on monitor port: %d", *queryPort)
		go listenAndServe(*queryPort)
	}

	select {}
}
