/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package main

import (
	compute "code.google.com/p/google-api-go-client/compute/v1"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/golang/glog"
	"golang.org/x/net/websocket"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/cloud/compute/metadata"
	"k8s.io/kubernetes/contrib/gce-firewall/common"
	"net/http"
	"time"
)

// stores the connection to a client
type ClientConnection struct {
	ws                 *websocket.Conn
	connTerminatedChan chan bool
}

// server configuration
type ConfigServer struct {
	// config has APIs to load/store Firewall rules
	firewallRulesStore common.ConfigInterface
	newClientChan      chan *ClientConnection
	// stores live connections to clients
	connections map[*ClientConnection]bool
	// server listens for web-socket connections on listenPort
	listenPort int
	flushChan  chan bool
	flush      bool
}

func main() {
	flagPort := flag.Int("port", 8080, "the port to listen on for websocket connections.")
	flag.Parse()

	config := common.NewFirewallConfig()
	configServer := &ConfigServer{
		firewallRulesStore: config,
		newClientChan:      make(chan *ClientConnection, 1000),
		connections:        make(map[*ClientConnection]bool),
		flushChan:          make(chan bool),
		listenPort:         *flagPort,
		flush:              false,
	}

	go configServer.readFirewallRulesInALoop()
	configServer.startWebSocketServer()
}

func (configServer *ConfigServer) readFirewallRulesInALoop() {
	ticker := time.NewTicker(10 * time.Second)
	for {
		select {
		case <-ticker.C:
			newConfigFound, _ := configServer.readConfig()
			if configServer.flush {
				configServer.pushConfigToAllClients([]*compute.Firewall{})
			} else {
				if newConfigFound {
					configServer.pushConfigToClients()
				}
			}
		case newConnection := <-configServer.newClientChan:
			configServer.connections[newConnection] = true
			if configServer.flush {
				configServer.pushConfigToAllClients([]*compute.Firewall{})
			} else {
				configServer.pushConfigToClients()
				glog.V(0).Infof("Client added to active connections list.")
			}
		case shouldFlush := <-configServer.flushChan:
			configServer.flush = shouldFlush
			if shouldFlush {
				configServer.pushConfigToAllClients([]*compute.Firewall{})
			} else {
				configServer.pushConfigToClients()
			}
		}
	}
}

func (configServer *ConfigServer) pushConfigToClients() {
	firewallRules, bytesToPush, updateNumber := configServer.firewallRulesStore.Load()
	if updateNumber == 0 {
		return
	}
	glog.V(4).Infof("Pushing config:%+s\n", string(bytesToPush))
	configServer.pushConfigToAllClients(firewallRules)
}

func (configServer *ConfigServer) pushConfigToAllClients(firewallRules []*compute.Firewall) {
	glog.V(0).Infof("Pushing %d firewall rules.", len(firewallRules))
	for connection := range configServer.connections {
		glog.V(0).Infof("Pushing to client %v\n", connection.ws.RemoteAddr())
		err := websocket.JSON.Send(connection.ws, firewallRules)
		if err != nil {
			fmt.Printf("Client died. %v\n", connection.ws.RemoteAddr())
			delete(configServer.connections, connection)
			close(connection.connTerminatedChan)
		}
	}
}

func (configServer *ConfigServer) readConfig() (bool, error) {
	rules := GetFirewallRules()
	stored, err := configServer.firewallRulesStore.Store(rules)
	if err != nil {
		glog.Errorf("Error serializing rules %v", err)
		return false, err
	}
	if !stored {
		glog.V(0).Infof("Rules have not changed %v", err)
	}
	return stored, nil
}

func (configServer *ConfigServer) startWebSocketServer() {
	http.Handle("/config", websocket.Handler(configServer.configRequestHandler))
	http.HandleFunc("/healthcheck", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "healthcheck: OK\n")
	})
	http.HandleFunc("/flush", func(w http.ResponseWriter, r *http.Request) {
		configServer.flushChan <- true
		fmt.Fprintf(w, "Flushing Firewall rules on all clients..\n")
	})
	http.HandleFunc("/start", func(w http.ResponseWriter, r *http.Request) {
		configServer.flushChan <- false
		fmt.Fprintf(w, "Stop Flushing Firewall rules on all clients and start pushing config\n")
	})

	err := http.ListenAndServe(fmt.Sprintf(":%d", configServer.listenPort), nil)
	if err != nil {
		panic("Error:" + err.Error())
	}
}

// invoked when a client requests for firewall rules
func (configServer *ConfigServer) configRequestHandler(ws *websocket.Conn) {
	terminate := make(chan bool, 1)
	clientConnection := &ClientConnection{
		ws:                 ws,
		connTerminatedChan: terminate,
	}
	configServer.newClientChan <- clientConnection

	select {
	case <-terminate:
		fmt.Printf("Client Connection closed.\n")
		return
	}
}

func GetFirewallRules() []*compute.Firewall {
	projectID, err := metadata.ProjectID()
	if err != nil {
		panic("Error:" + err.Error())
	}

	tokenSource := google.ComputeTokenSource("")
	client := oauth2.NewClient(oauth2.NoContext, tokenSource)
	gCompute, err := compute.New(client)
	if err != nil {
		panic("Error:" + err.Error())
	}

	listCall := gCompute.Firewalls.List(projectID)
	res, err := listCall.Do()
	if err != nil {
		panic("Error:" + err.Error())
	}
	serialized, err := json.Marshal(res.Items)
	if err == nil {
		glog.V(4).Infof("Firewall Rules: %+v", string(serialized))
	}
	return res.Items
}
