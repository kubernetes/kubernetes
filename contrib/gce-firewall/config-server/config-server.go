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
	"flag"
	"fmt"
	"github.com/GoogleCloudPlatform/kubernetes/contrib/gce-firewall/common"
	"github.com/golang/glog"
	"golang.org/x/net/websocket"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/cloud/compute/metadata"
	"net/http"
	"time"
)

type ClientConnection struct {
	ws        *websocket.Conn
	terminate chan bool
}

type ConfigServer struct {
	config      common.ConfigInterface
	registry    chan *ClientConnection
	connections map[*ClientConnection]bool
	listenPort  int
}

func main() {
	flagPort := flag.Int("port", 8080, "the port to listen on.")
	flag.Parse()

	config := common.NewFirewallConfig()
	configServer := &ConfigServer{
		config:      config,
		registry:    make(chan *ClientConnection, 1000),
		connections: make(map[*ClientConnection]bool),
		listenPort:  *flagPort,
	}

	go configServer.readFirewallRulesInALoop()
	configServer.startWebSocketServer()
}

func (configServer *ConfigServer) readFirewallRulesInALoop() {
	ticker := time.NewTicker(10 * time.Second)
	for {
		select {
		case <-ticker.C:
			configServer.readConfigAndPushToClients()
		case newConnection := <-configServer.registry:
			configServer.connections[newConnection] = true
			configServer.pushConfigToClients()
			fmt.Printf("Client added to active connections list.\n")
		}
	}
}

func (configServer *ConfigServer) pushConfigToClients() {
	firewallRules, bytesToPush, updateNumber := configServer.config.Load()
	if updateNumber == 0 {
		return
	}
	for connection := range configServer.connections {
		fmt.Printf("Pushing to client %v. config:%+s\n", connection.ws.RemoteAddr(), string(bytesToPush))
		err := websocket.JSON.Send(connection.ws, firewallRules)
		if err != nil {
			fmt.Printf("Client died. %v\n", connection.ws.RemoteAddr())
			delete(configServer.connections, connection)
			close(connection.terminate)
		}
	}
}

func (configServer *ConfigServer) readConfigAndPushToClients() {
	rules := GetFirewallRules()
	stored, err := configServer.config.Store(rules)
	if err != nil {
		glog.Errorf("Error serializing rules %v", err)
		return
	}
	if !stored {
		glog.V(0).Infof("Rules have not changed %v", err)
		return
	}
	configServer.pushConfigToClients()
}

func (configServer *ConfigServer) startWebSocketServer() {
	http.Handle("/config", websocket.Handler(configServer.configHandler))
	err := http.ListenAndServe(fmt.Sprintf(":%d", configServer.listenPort), nil)
	if err != nil {
		panic("Error:" + err.Error())
	}
}

func (configServer *ConfigServer) configHandler(ws *websocket.Conn) {
	terminate := make(chan bool, 1)
	clientConnection := &ClientConnection{
		ws:        ws,
		terminate: terminate,
	}
	configServer.registry <- clientConnection

	select {
	case <-terminate:
		fmt.Printf("Client Connection closed.\n")
		return
	}
}

func GetFirewallRules() []*compute.Firewall {
	//	tmp := make([]*compute.Firewall, 1)
	//	allowed := make([]*compute.FirewallAllowed, 1)
	//	allowed[0] = &compute.FirewallAllowed{
	//		IPProtocol: "http",
	//		Ports:      []string{"80", "8000-8001"},
	//	}
	//	tmp[0] = &compute.Firewall{
	//		Description: "FooDescription",
	//		Allowed:     allowed,
	//	}
	//	return tmp
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
	glog.Infof("Firewall Rules: %+v", res.Items)
	return res.Items
}
