/*
Copyright 2015 The Kubernetes Authors.

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

package testing

import (
	"io/ioutil"
	"net"
	"os"
	"strconv"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/storage/consul/consultest"
	"k8s.io/kubernetes/pkg/storage/storagebackend"

	consulapi "github.com/hashicorp/consul/api"
	"github.com/hashicorp/consul/command/agent"
	"github.com/hashicorp/consul/consul"
)

type ConsulTestClientServer struct {
	config          storagebackend.Config
	Client          *consulapi.Client
	ClientConfig    *consulapi.Config
	serverConfig    *consul.Config
	server          *consul.Server
	CertificatesDir string
	ConfigFile      string
	serverReady     chan bool
	agent           *agent.Agent
	httpservers     []*agent.HTTPServer
}

func NewConsulTestClientServer(t *testing.T) *ConsulTestClientServer {
	t.Logf("Starting Consul")
	server := &ConsulTestClientServer{}

	server.serverReady = make(chan bool)

	lRPC, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("%s", err)
	}

	lHTTP, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("%s", err)
	}

	lServer, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("%s", err)
	}

	lSerfLAN, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("%s", err)
	}

	lSerfWAN, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("%s", err)
	}

	lRPCAddr := lRPC.Addr().String()
	err = lRPC.Close()
	if err != nil {
		t.Fatalf("%s", err)
	}

	lHTTPAddr := lHTTP.Addr().String()
	err = lHTTP.Close()
	if err != nil {
		t.Fatalf("%s", err)
	}

	lServerAddr := lServer.Addr().String()
	err = lServer.Close()
	if err != nil {
		t.Fatalf("%s", err)
	}

	lSerfLANAddr := lSerfLAN.Addr().String()
	err = lSerfLAN.Close()
	if err != nil {
		t.Fatalf("%s", err)
	}

	lSerfWANAddr := lSerfWAN.Addr().String()
	err = lSerfWAN.Close()
	if err != nil {
		t.Fatalf("%s", err)
	}

	server.config = storagebackend.Config{
		Type:       storagebackend.StorageTypeConsul,
		ServerList: []string{"http://" + lHTTPAddr},
		Prefix:     consultest.PathPrefix(),
	}

	//get tmp dir
	tmpDir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Failed to create temp dir for consul")
	}

	agentConfig := agent.DevConfig()
	agentConfig.Server = true
	agentConfig.DataDir = tmpDir
	agentConfig.Ports.HTTPS = -1
	agentConfig.Ports.DNS = -1

	_, port, err := net.SplitHostPort(lHTTPAddr)
	if err != nil {
		t.Fatalf("%s", err)
	}
	agentConfig.Ports.HTTP, err = strconv.Atoi(port)
	if err != nil {
		t.Fatalf("%s", err)
	}

	_, port, err = net.SplitHostPort(lRPCAddr)
	if err != nil {
		t.Fatalf("%s", err)
	}
	agentConfig.Ports.RPC, err = strconv.Atoi(port)
	if err != nil {
		t.Fatalf("%s", err)
	}

	_, port, err = net.SplitHostPort(lServerAddr)
	if err != nil {
		t.Fatalf("%s", err)
	}
	agentConfig.Ports.Server, err = strconv.Atoi(port)
	if err != nil {
		t.Fatalf("%s", err)
	}

	_, port, err = net.SplitHostPort(lSerfLANAddr)
	if err != nil {
		t.Fatalf("%s", err)
	}
	agentConfig.Ports.SerfLan, err = strconv.Atoi(port)
	if err != nil {
		t.Fatalf("%s", err)
	}

	_, port, err = net.SplitHostPort(lSerfWANAddr)
	if err != nil {
		t.Fatalf("%s", err)
	}
	agentConfig.Ports.SerfWan, err = strconv.Atoi(port)
	if err != nil {
		t.Fatalf("%s", err)
	}

	var consulAgent *agent.Agent
	attempts := 0
	//try to start Consul
	for {
		if attempts > 5 {
			t.Fatalf("Tried to start Consul %d times without success", attempts)
		}
		attempts++

		consulAgent, err = agent.Create(agentConfig, nil)
		if err != nil {
			t.Logf("Failed to launch Consul: %s", err)
		} else {
			//agent is up... continue
			break
		}
		time.Sleep(250 * time.Millisecond)
	}

	server.agent = consulAgent

	//start the HTTP server manually
	server.httpservers, err = agent.NewHTTPServers(consulAgent, agentConfig, os.Stdout)
	if err != nil {
		t.Fatalf("Failed to start http server: %+v", err)
	}

	clientConfig := consulapi.DefaultConfig()
	clientConfig.Address = lHTTP.Addr().String()

	client, err := consulapi.NewClient(clientConfig)
	if err != nil {
		t.Fatalf("Failed to create Consul client: %s", err)
	}

	server.Client = client
	server.ClientConfig = clientConfig

	//wait until election etc is done
	//TODO: a sane more reliable readyness check
	time.Sleep(time.Duration(10) * time.Second)

	return server
}

func (s *ConsulTestClientServer) Terminate(t *testing.T) {
	t.Logf("Terminating Consul")
	err := s.agent.Leave()
	if err != nil {
		t.Log("Error leaving Consul: %s", err)
	}

	for _, s := range s.httpservers {
		s.Shutdown()
	}

	err = s.agent.Shutdown()
	if err != nil {
		t.Fatalf("Failed to stop Consul: %v", err)
	}

	t.Logf("Waiting for Consul to shutdown...")
	<-s.agent.ShutdownCh()
	time.Sleep(500 * time.Millisecond)
	t.Logf("Consul shut down complete!")
}
