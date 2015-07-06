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
	ws "golang.org/x/net/websocket"
	"time"
)

const (
	origin = "http://localhost/"
)

type ConfigClient struct {
	serverEndpoint      string
	config              common.ConfigInterface
	appliedUpdateNumber int
}

func main() {
	flagServer := flag.String("server", "localhost", "service name that provides the GCE Firewall Rules")
	flagPort := flag.Int("port", 8080, "port of service that provides the GCE Firewall Rules")
	flag.Parse()

	client := &ConfigClient{
		serverEndpoint:      fmt.Sprintf("ws://%s:%d/config", *flagServer, *flagPort),
		config:              common.NewFirewallConfig(),
		appliedUpdateNumber: 0,
	}
	client.mainLoop()
}

func (client *ConfigClient) mainLoop() {
	for {
		conn := client.createConnection()
		client.applyConfigLoop(conn)
	}
}

func (client *ConfigClient) createConnection() *ws.Conn {
	glog.V(0).Infof("Connecting to %s", client.serverEndpoint)

	ticker := time.NewTicker(1 * time.Second)
	for {
		select {
		case <-ticker.C:
			conn, err := ws.Dial(client.serverEndpoint, "", origin)
			if err == nil {
				glog.V(0).Infof("Connection established with %s", client.serverEndpoint)
				return conn
			}
		}
	}
}

func (client *ConfigClient) applyConfigLoop(conn *ws.Conn) {
	firewallRulesChannel := make(chan bool, 1000)
	go client.poulateFirewallRulesChannel(conn, firewallRulesChannel)

	ticker := time.NewTicker(1 * time.Second)
	for {
		select {
		case <-ticker.C:
			client.ensureRulesAreApplied()
		case _, more := <-firewallRulesChannel:
			client.ensureRulesAreApplied()
			if !more {
				return
			}
		}
	}
}

func (client *ConfigClient) ensureRulesAreApplied() {
	firewallRules, serialized, updateNumber := client.config.Load()
	if client.appliedUpdateNumber < updateNumber {
		glog.V(0).Infof("Applied firewall rules %s", string(serialized))
		err := applyFirewallRules(firewallRules)
		if err != nil {
			glog.Errorf("Could not apply firewall rules. Wait for next timer loop. Still on update:%d. Error:%v", client.appliedUpdateNumber, err)
		} else {
			glog.Errorf("Applied firewall rules. updateNumber:%d", updateNumber)
			client.appliedUpdateNumber = updateNumber
		}
	}
}

func (client *ConfigClient) poulateFirewallRulesChannel(conn *ws.Conn, firewallRulesChannel chan bool) {
	for {
		var firewallRules []*compute.Firewall
		err := ws.JSON.Receive(conn, &firewallRules)
		if err != nil {
			glog.Errorf("Could not read from config-server. %+v", err)
			close(firewallRulesChannel)
			return
		}
		_, err = client.config.Store(firewallRules)
		if err != nil {
			glog.Errorf("Could not store config: err: %+v", err)
		} else {
			firewallRulesChannel <- true
		}
	}
}
