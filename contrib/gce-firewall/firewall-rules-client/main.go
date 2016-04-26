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
	"github.com/golang/glog"
	ws "golang.org/x/net/websocket"
	"k8s.io/kubernetes/contrib/gce-firewall/common"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	origin = "http://localhost/"
)

// client configuration
type FirewallClient struct {
	firewallServerEndpoint string
	firewallRulesStore     common.ConfigInterface
	appliedUpdateNumber    int
}

func main() {
	client := createFirewallClient()
	client.mainLoop()
}

func createFirewallClient() *FirewallClient {
	flagServiceHost := flag.String("serviceHost", "", "Service host that provides the GCE Firewall Rules")
	flagServicePort := flag.Int("servicePort", 0, "Service Port that provides the GCE Firewall Rules")
	flagServiceName := flag.String("serviceName", "", "Service name that provices the GCE Firewall Rules. Used only if server and port are not specified.")

	flag.Parse()
	serviceHost := *flagServiceHost
	servicePort := *flagServicePort
	serviceName := *flagServiceName

	if serviceHost == "" && serviceName == "" {
		glog.Errorf("serviceHost or serviceName should be specified.")
		os.Exit(1)
	}

	if serviceName != "" {
		r := regexp.MustCompile("-")
		serviceName = r.ReplaceAllString(strings.ToUpper(serviceName), "_")
	}

	glog.Error("Service Name is " + serviceName)

	// server was not specified, read the Service IP and Port from the environment variable
	if serviceHost == "" {
		envVar := serviceName + "_SERVICE_HOST"
		serviceHost = os.Getenv(envVar)
		if serviceHost == "" {
			glog.Errorf("Service Host could not be derived from env var:%q", envVar)
			os.Exit(1)
		}
	}

	if servicePort == 0 {
		var err error
		envVar := serviceName + "_SERVICE_PORT"
		servicePort, err = strconv.Atoi(os.Getenv(envVar))
		if err != nil {
			glog.Errorf("Service Port could not be derived from envVar: %s", envVar)
			os.Exit(1)
		}
	}

	if servicePort == 0 {
		glog.Errorf("Service Port should be specified and should be non-zero.")
		os.Exit(1)
	}

	client := &FirewallClient{
		firewallServerEndpoint: fmt.Sprintf("ws://%s:%d/config", serviceHost, servicePort),
		firewallRulesStore:     common.NewFirewallConfig(),
		appliedUpdateNumber:    0,
	}

	return client
}

func (client *FirewallClient) mainLoop() {
	for {
		conn := client.createConnection()
		client.applyFirewallRulesLoop(conn)
	}
}

func (client *FirewallClient) createConnection() *ws.Conn {
	glog.V(0).Infof("Connecting to %s", client.firewallServerEndpoint)

	ticker := time.NewTicker(1 * time.Second)
	for {
		select {
		case <-ticker.C:
			conn, err := ws.Dial(client.firewallServerEndpoint, "", origin)
			if err == nil {
				glog.V(0).Infof("Connection established with %s", client.firewallServerEndpoint)
				return conn
			} else {
				glog.Errorf("Connection could not be established to %s, err:%v", client.firewallServerEndpoint, err)
			}
		}
	}
}

func (client *FirewallClient) applyFirewallRulesLoop(conn *ws.Conn) {
	firewallRulesChannel := make(chan bool, 1000)
	go client.fetchFirewallRules(conn, firewallRulesChannel)

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

func (client *FirewallClient) ensureRulesAreApplied() {
	firewallRules, serialized, updateNumber := client.firewallRulesStore.Load()
	if client.appliedUpdateNumber < updateNumber {
		glog.V(0).Infof("Applying firewall rules %s", string(serialized))
		err := applyFirewallRules(firewallRules)
		if err != nil {
			glog.Errorf("Could not apply firewall rules. Wait for next timer loop. Still on update:%d. Error:%v", client.appliedUpdateNumber, err)
		} else {
			glog.Errorf("Applied firewall rules. updateNumber:%d", updateNumber)
			client.appliedUpdateNumber = updateNumber
		}
	}
}

// fetches firewall rules from server and sends a notification on the firewallRulesChan channel
func (client *FirewallClient) fetchFirewallRules(conn *ws.Conn, firewallRulesChan chan bool) {
	for {
		var firewallRules []*compute.Firewall
		err := ws.JSON.Receive(conn, &firewallRules)
		if err != nil {
			glog.Errorf("Could not read from firewall-rules-server. %+v", err)
			close(firewallRulesChan)
			return
		}
		_, err = client.firewallRulesStore.Store(firewallRules)
		if err != nil {
			glog.Errorf("Could not store config: err: %+v", err)
		} else {
			glog.V(0).Infof("Read Firewall Rules. Count: %d", len(firewallRules))
			firewallRulesChan <- true
		}
	}
}
