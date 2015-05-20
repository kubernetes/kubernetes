/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// A tiny web server for checking networking connectivity.
//
// Will dial out to, and expect to hear from, every pod that is a member of
// the service passed in the flag -service.
//
// Will serve a webserver on given -port.
//
// Visit /read to see the current state, or /quit to shut down.
//
// Visit /status to see pass/running/fail determination. (literally, it will
// return one of those words.)
//
// /write is used by other network test pods to register connectivity.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

var (
	port      = flag.Int("port", 8080, "Port number to serve at.")
	peerCount = flag.Int("peers", 8, "Must find at least this many peers for the test to pass.")
	service   = flag.String("service", "nettest", "Service to find other network test pods in.")
	namespace = flag.String("namespace", "default", "Namespace of this pod. TODO: kubernetes should make this discoverable.")
)

// State tracks the internal state of our little http server.
// It's returned verbatim over the /read endpoint.
type State struct {
	// Hostname is set once and never changed-- it's always safe to read.
	Hostname string

	// The below fields require that lock is held before reading or writing.
	Sent                 map[string]int
	Received             map[string]int
	Errors               []string
	Log                  []string
	StillContactingPeers bool

	lock sync.Mutex
}

func (s *State) doneContactingPeers() {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.StillContactingPeers = false
}

// serveStatus returns "pass", "running", or "fail".
func (s *State) serveStatus(w http.ResponseWriter, r *http.Request) {
	s.lock.Lock()
	defer s.lock.Unlock()
	if len(s.Sent) >= *peerCount && len(s.Received) >= *peerCount {
		fmt.Fprintf(w, "pass")
		return
	}
	if s.StillContactingPeers {
		fmt.Fprintf(w, "running")
		return
	}
	s.Logf("Declaring failure for %s/%s with %d sent and %d received and %d peers", *namespace, *service, s.Sent, s.Received, *peerCount)
	fmt.Fprintf(w, "fail")
}

// serveRead writes our json encoded state
func (s *State) serveRead(w http.ResponseWriter, r *http.Request) {
	s.lock.Lock()
	defer s.lock.Unlock()
	w.WriteHeader(http.StatusOK)
	b, err := json.MarshalIndent(s, "", "\t")
	s.appendErr(err)
	_, err = w.Write(b)
	s.appendErr(err)
}

// WritePost is the format that (json encoded) requests to the /write handler should take.
type WritePost struct {
	Source string
	Dest   string
}

// WriteResp is returned by /write
type WriteResp struct {
	Hostname string
}

// serveWrite records the contact in our state.
func (s *State) serveWrite(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	s.lock.Lock()
	defer s.lock.Unlock()
	w.WriteHeader(http.StatusOK)
	var wp WritePost
	s.appendErr(json.NewDecoder(r.Body).Decode(&wp))
	if wp.Source == "" {
		s.appendErr(fmt.Errorf("%v: Got request with no source", s.Hostname))
	} else {
		if s.Received == nil {
			s.Received = map[string]int{}
		}
		s.Received[wp.Source] += 1
	}
	s.appendErr(json.NewEncoder(w).Encode(&WriteResp{Hostname: s.Hostname}))
}

// appendErr adds err to the list, if err is not nil. s must be locked.
func (s *State) appendErr(err error) {
	if err != nil {
		s.Errors = append(s.Errors, err.Error())
	}
}

// Logf writes to the log message list. s must not be locked.
// s's Log member will drop an old message if it would otherwise
// become longer than 500 messages.
func (s *State) Logf(format string, args ...interface{}) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.Log = append(s.Log, fmt.Sprintf(format, args...))
	if len(s.Log) > 500 {
		s.Log = s.Log[1:]
	}
}

// s must not be locked
func (s *State) appendSuccessfulSend(toHostname string) {
	s.lock.Lock()
	defer s.lock.Unlock()
	if s.Sent == nil {
		s.Sent = map[string]int{}
	}
	s.Sent[toHostname] += 1
}

var (
	// Our one and only state object
	state State
)

func main() {
	flag.Parse()

	if *service == "" {
		log.Fatal("Must provide -service flag.")
	}

	hostname, err := os.Hostname()
	if err != nil {
		log.Fatalf("Error getting hostname: %v", err)
	}

	state := State{
		Hostname:             hostname,
		StillContactingPeers: true,
	}

	go contactOthers(&state)

	http.HandleFunc("/quit", func(w http.ResponseWriter, r *http.Request) {
		os.Exit(0)
	})

	http.HandleFunc("/read", state.serveRead)
	http.HandleFunc("/write", state.serveWrite)
	http.HandleFunc("/status", state.serveStatus)

	go log.Fatal(http.ListenAndServe(fmt.Sprintf("0.0.0.0:%d", *port), nil))

	select {}
}

// Find all sibling pods in the service and post to their /write handler.
func contactOthers(state *State) {
	defer state.doneContactingPeers()
	masterRO := url.URL{
		Scheme: "http",
		Host:   os.Getenv("KUBERNETES_RO_SERVICE_HOST") + ":" + os.Getenv("KUBERNETES_RO_SERVICE_PORT"),
		Path:   "/api/" + latest.Version,
	}
	client := &client.Client{client.NewRESTClient(&masterRO, latest.Version, latest.Codec, false, 5, 10)}

	// Do this repeatedly, in case there's some propagation delay with getting
	// newly started pods into the endpoints list.
	for i := 0; i < 15; i++ {
		endpoints, err := client.Endpoints(*namespace).Get(*service)
		if err != nil {
			state.Logf("Unable to read the endpoints for %v/%v: %v; will try again.", *namespace, *service, err)
			time.Sleep(time.Duration(1+rand.Intn(10)) * time.Second)
		}

		eps := util.StringSet{}
		for _, ss := range endpoints.Subsets {
			for _, a := range ss.Addresses {
				for _, p := range ss.Ports {
					eps.Insert(fmt.Sprintf("http://%s:%d", a.IP, p.Port))
				}
			}
		}
		for ep := range eps {
			state.Logf("Attempting to contact %s", ep)
			contactSingle(ep, state)
		}

		time.Sleep(5 * time.Second)
	}
}

// contactSingle dials the address 'e' and tries to POST to its /write address.
func contactSingle(e string, state *State) {
	body, err := json.Marshal(&WritePost{
		Dest:   e,
		Source: state.Hostname,
	})
	if err != nil {
		log.Fatalf("json marshal error: %v", err)
	}
	resp, err := http.Post(e+"/write", "application/json", bytes.NewReader(body))
	if err != nil {
		state.Logf("Warning: unable to contact the endpoint %q: %v", e, err)
		return
	}
	defer resp.Body.Close()

	body, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		state.Logf("Warning: unable to read response from '%v': '%v'", e, err)
		return
	}
	var wr WriteResp
	err = json.Unmarshal(body, &wr)
	if err != nil {
		state.Logf("Warning: unable to unmarshal response (%v) from '%v': '%v'", string(body), e, err)
		return
	}
	state.appendSuccessfulSend(wr.Hostname)
}
