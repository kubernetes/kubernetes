/*
Copyright 2014 The Kubernetes Authors.

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
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
)

var (
	port          = flag.Int("port", 8080, "Port number to serve at.")
	peerCount     = flag.Int("peers", 8, "Must find at least this many peers for the test to pass.")
	service       = flag.String("service", "nettest", "Service to find other network test pods in.")
	namespace     = flag.String("namespace", "default", "Namespace of this pod. TODO: kubernetes should make this discoverable.")
	delayShutdown = flag.Int("delay-shutdown", 0, "Number of seconds to delay shutdown when receiving SIGTERM.")
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
	// Logf can't be called while holding the lock, so defer using a goroutine
	go s.Logf("Declaring failure for %s/%s with %d sent and %d received and %d peers", *namespace, *service, len(s.Sent), len(s.Received), *peerCount)
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

	if *delayShutdown > 0 {
		termCh := make(chan os.Signal)
		signal.Notify(termCh, syscall.SIGTERM)
		go func() {
			<-termCh
			log.Printf("Sleeping %d seconds before exit ...", *delayShutdown)
			time.Sleep(time.Duration(*delayShutdown) * time.Second)
			os.Exit(0)
		}()
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
	sleepTime := 5 * time.Second
	// In large cluster getting all endpoints is pretty expensive.
	// Thus, we will limit ourselves to send on average at most 10 such
	// requests per second
	if sleepTime < time.Duration(*peerCount/10)*time.Second {
		sleepTime = time.Duration(*peerCount/10) * time.Second
	}
	timeout := 5 * time.Minute
	// Similarly we need to bump timeout so that it is reasonable in large
	// clusters.
	if timeout < time.Duration(*peerCount)*time.Second {
		timeout = time.Duration(*peerCount) * time.Second
	}
	defer state.doneContactingPeers()

	config, err := restclient.InClusterConfig()
	if err != nil {
		log.Fatalf("Unable to create config; error: %v\n", err)
	}
	config.ContentType = "application/vnd.kubernetes.protobuf"
	client, err := clientset.NewForConfig(config)
	if err != nil {
		log.Fatalf("Unable to create client; error: %v\n", err)
	}
	// Double check that that worked by getting the server version.
	if v, err := client.Discovery().ServerVersion(); err != nil {
		log.Fatalf("Unable to get server version: %v\n", err)
	} else {
		log.Printf("Server version: %#v\n", v)
	}

	for start := time.Now(); time.Since(start) < timeout; time.Sleep(sleepTime) {
		eps := getWebserverEndpoints(client)
		if eps.Len() >= *peerCount {
			break
		}
		state.Logf("%v/%v has %v endpoints (%v), which is less than %v as expected. Waiting for all endpoints to come up.", *namespace, *service, len(eps), eps.List(), *peerCount)
	}

	// Do this repeatedly, in case there's some propagation delay with getting
	// newly started pods into the endpoints list.
	for i := 0; i < 15; i++ {
		eps := getWebserverEndpoints(client)
		for ep := range eps {
			state.Logf("Attempting to contact %s", ep)
			contactSingle(ep, state)
		}
		time.Sleep(sleepTime)
	}
}

//getWebserverEndpoints returns the webserver endpoints as a set of String, each in the format like "http://{ip}:{port}"
func getWebserverEndpoints(client clientset.Interface) sets.String {
	endpoints, err := client.Core().Endpoints(*namespace).Get(*service, v1.GetOptions{})
	eps := sets.String{}
	if err != nil {
		state.Logf("Unable to read the endpoints for %v/%v: %v.", *namespace, *service, err)
		return eps
	}
	for _, ss := range endpoints.Subsets {
		for _, a := range ss.Addresses {
			for _, p := range ss.Ports {
				eps.Insert(fmt.Sprintf("http://%s:%d", a.IP, p.Port))
			}
		}
	}
	return eps
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
