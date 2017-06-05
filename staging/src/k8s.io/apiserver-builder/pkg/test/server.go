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

package test

import (
	"bufio"
	"fmt"
	"io"
	"io/ioutil"
	openapi "k8s.io/apimachinery/pkg/openapi"
	"k8s.io/apiserver-builder/pkg/builders"
	"k8s.io/apiserver-builder/pkg/cmd/server"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/rest"
	"net"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

type TestEnvironment struct {
	StopServer     chan struct{}
	ServerOuput    *io.PipeWriter
	ApiserverPort  int
	BearerToken    string
	EtcdClientPort int
	EtcdPeerPort   int
	EtcdPath       string
	EtcdCmd        *exec.Cmd
	Done           bool
}

func NewTestEnvironment() *TestEnvironment {
	return &TestEnvironment{
		EtcdPath: "/registry/test.kubernetes.io",
	}
}

func (te *TestEnvironment) getPort() int {
	l, _ := net.Listen("tcp", ":0")
	defer l.Close()
	println(l.Addr().String())
	pieces := strings.Split(l.Addr().String(), ":")
	i, err := strconv.Atoi(pieces[len(pieces)-1])
	if err != nil {
		panic(err)
	}
	return i
}

// Stop stops a running server
func (te *TestEnvironment) Stop() {
	te.Done = true
	te.StopServer <- struct{}{}
	te.EtcdCmd.Process.Kill()
}

// Start starts a local Kubernetes server and updates te.ApiserverPort with the port it is listening on
func (te *TestEnvironment) Start(
	apis []*builders.APIGroupBuilder, openapidefs openapi.GetOpenAPIDefinitions) *rest.Config {

	te.EtcdClientPort = te.getPort()
	te.EtcdPeerPort = te.getPort()
	te.ApiserverPort = te.getPort()

	etcdready := make(chan string)
	go te.startEtcd(etcdready)

	apiserverready := make(chan *rest.Config)
	go te.startApiserver(apiserverready, apis, openapidefs)

	// Wait for everything to be ready
	loopback := <-apiserverready
	<-etcdready
	return loopback
}

func (te *TestEnvironment) startApiserver(
	ready chan *rest.Config, apis []*builders.APIGroupBuilder, openapidefs openapi.GetOpenAPIDefinitions) {
	te.StopServer = make(chan struct{})
	_, te.ServerOuput = io.Pipe()
	server.GetOpenApiDefinition = openapidefs
	cmd, options := server.NewCommandStartServer(
		te.EtcdPath,
		te.ServerOuput, te.ServerOuput, apis, te.StopServer)

	options.RecommendedOptions.SecureServing.BindPort = te.ApiserverPort
	options.RunDelegatedAuth = false
	options.RecommendedOptions.Etcd.StorageConfig.ServerList = []string{
		fmt.Sprintf("http://localhost:%d", te.EtcdClientPort),
	}

	// Notify once the apiserver is ready to serve traffic
	options.PostStartHooks = []server.PostStartHook{
		{
			func(context genericapiserver.PostStartHookContext) error {
				// Let the test know the server is ready
				ready <- context.LoopbackClientConfig
				return nil
			},
			"apiserver-ready",
		},
	}

	if err := cmd.Execute(); err != nil {
		panic(err)
	}
}

// startEtcd starts a new etcd process using a random temp data directory and random free port
func (te *TestEnvironment) startEtcd(ready chan string) {
	dirname, err := ioutil.TempDir("/tmp", "apiserver-test")
	if err != nil {
		panic(err)
	}

	clientAddr := fmt.Sprintf("http://localhost:%d", te.EtcdClientPort)
	peerAddr := fmt.Sprintf("http://localhost:%d", te.EtcdPeerPort)
	cmd := exec.Command(
		"etcd",
		"--data-dir", dirname,
		"--listen-client-urls", clientAddr,
		"--listen-peer-urls", peerAddr,
		"--advertise-client-urls", clientAddr,
	)
	te.EtcdCmd = cmd
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		panic(err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		panic(err)
	}

	err = cmd.Start()
	if err != nil {
		panic(err)
	}

	go te.waitForEtcdReady(ready, stdout)
	go te.waitForEtcdReady(ready, stderr)

	err = cmd.Wait()
	if err != nil && !te.Done {
		panic(err)
	}
}

// waitForEtcdReady notify's read once the etcd instances is ready to receive traffic
func (te *TestEnvironment) waitForEtcdReady(ready chan string, reader io.Reader) {
	started := regexp.MustCompile("serving insecure client requests on (.+), this is strongly discouraged!")
	buffered := bufio.NewReader(reader)
	for {
		l, _, err := buffered.ReadLine()
		if err != nil {
			time.Sleep(time.Second * 5)
		}
		line := string(l)
		if started.MatchString(line) {
			addr := started.FindStringSubmatch(line)[1]
			// etcd is ready
			ready <- addr
			return
		}
	}
}
