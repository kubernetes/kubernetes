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

package mesos

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"testing"
	"time"

	log "github.com/golang/glog"
	"github.com/mesos/mesos-go/detector"
	"github.com/mesos/mesos-go/mesosutil"
	"golang.org/x/net/context"

	utilnet "k8s.io/kubernetes/pkg/util/net"
)

// Test data

const (
	TEST_MASTER_ID   = "master-12345"
	TEST_MASTER_IP   = 177048842 // 10.141.141.10
	TEST_MASTER_PORT = 5050

	TEST_STATE_JSON = `
	{
		"version": "0.22.0",
		"unregistered_frameworks": [],
		"started_tasks": 0,
		"start_time": 1429456501.61141,
		"staged_tasks": 0,
		"slaves": [
		{
			"resources": {
				"ports": "[31000-32000]",
				"mem": 15360,
				"disk": 470842,
				"cpus": 8
			},
			"registered_time": 1429456502.46999,
			"pid": "slave(1)@mesos1.internal.example.org.fail:5050",
			"id": "20150419-081501-16777343-5050-16383-S2",
			"hostname": "mesos1.internal.example.org.fail",
			"attributes": {},
			"active": true
		},
		{
			"resources": {
				"ports": "[31000-32000]",
				"mem": 15360,
				"disk": 470842,
				"cpus": 8
			},
			"registered_time": 1429456502.4144,
			"pid": "slave(1)@mesos2.internal.example.org.fail:5050",
			"id": "20150419-081501-16777343-5050-16383-S1",
			"hostname": "mesos2.internal.example.org.fail",
			"attributes": {},
			"active": true
		},
		{
			"resources": {
				"ports": "[31000-32000]",
				"mem": 15360,
				"disk": 470842,
				"cpus": 8
			},
			"registered_time": 1429456502.02879,
			"pid": "slave(1)@mesos3.internal.example.org.fail:5050",
			"id": "20150419-081501-16777343-5050-16383-S0",
			"hostname": "mesos3.internal.example.org.fail",
			"attributes": {},
			"active": true
		}
		],
		"pid": "master@mesos-master0.internal.example.org.fail:5050",
		"orphan_tasks": [],
		"lost_tasks": 0,
		"leader": "master@mesos-master0.internal.example.org.fail:5050",
		"killed_tasks": 0,
		"failed_tasks": 0,
		"elected_time": 1429456501.61638,
		"deactivated_slaves": 0,
		"completed_frameworks": [],
		"build_user": "buildbot",
		"build_time": 1425085311,
		"build_date": "2015-02-27 17:01:51",
		"activated_slaves": 3,
		"finished_tasks": 0,
		"flags": {
			"zk_session_timeout": "10secs",
			"work_dir": "/somepath/mesos/local/Lc9arz",
			"webui_dir": "/usr/local/share/mesos/webui",
			"version": "false",
			"user_sorter": "drf",
			"slave_reregister_timeout": "10mins",
			"logbufsecs": "0",
			"log_auto_initialize": "true",
			"initialize_driver_logging": "true",
			"framework_sorter": "drf",
			"authenticators": "crammd5",
			"authenticate_slaves": "false",
			"authenticate": "false",
			"allocation_interval": "1secs",
			"logging_level": "INFO",
			"quiet": "false",
			"recovery_slave_removal_limit": "100%",
			"registry": "replicated_log",
			"registry_fetch_timeout": "1mins",
			"registry_store_timeout": "5secs",
			"registry_strict": "false",
			"root_submissions": "true"
		},
		"frameworks": [],
		"git_branch": "refs/heads/0.22.0-rc1",
		"git_sha": "46834faca67f877631e1beb7d61be5c080ec3dc2",
		"git_tag": "0.22.0-rc1",
		"hostname": "localhost",
		"id": "20150419-081501-16777343-5050-16383"
	}`
)

// Mocks

type FakeMasterDetector struct {
	callback detector.MasterChanged
	done     chan struct{}
}

func newFakeMasterDetector() *FakeMasterDetector {
	return &FakeMasterDetector{
		done: make(chan struct{}),
	}
}

func (md FakeMasterDetector) Cancel() {
	close(md.done)
}

func (md FakeMasterDetector) Detect(cb detector.MasterChanged) error {
	md.callback = cb
	leadingMaster := mesosutil.NewMasterInfo(TEST_MASTER_ID, TEST_MASTER_IP, TEST_MASTER_PORT)
	cb.OnMasterChanged(leadingMaster)
	return nil
}

func (md FakeMasterDetector) Done() <-chan struct{} {
	return md.done
}

// Auxiliary functions

func makeHttpMocks() (*httptest.Server, *http.Client, *http.Transport) {
	httpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.V(4).Infof("Mocking response for HTTP request: %#v", r)
		if r.URL.Path == "/state.json" {
			w.WriteHeader(200) // OK
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintln(w, TEST_STATE_JSON)
		} else {
			w.WriteHeader(400)
			fmt.Fprintln(w, "Bad Request")
		}
	}))

	// Intercept all client requests and feed them to the test server
	transport := utilnet.SetTransportDefaults(&http.Transport{
		Proxy: func(req *http.Request) (*url.URL, error) {
			return url.Parse(httpServer.URL)
		},
	})

	httpClient := &http.Client{Transport: transport}

	return httpServer, httpClient, transport
}

// Tests

// test mesos.parseMesosState
func Test_parseMesosState(t *testing.T) {
	state, err := parseMesosState([]byte(TEST_STATE_JSON))

	if err != nil {
		t.Fatalf("parseMesosState does not yield an error")
	}
	if state == nil {
		t.Fatalf("parseMesosState yields a non-nil state")
	}
	if len(state.nodes) != 3 {
		t.Fatalf("parseMesosState yields a state with 3 nodes")
	}
}

// test mesos.listSlaves
func Test_listSlaves(t *testing.T) {
	defer log.Flush()
	md := FakeMasterDetector{}
	httpServer, httpClient, httpTransport := makeHttpMocks()
	defer httpServer.Close()

	cacheTTL := 500 * time.Millisecond
	mesosClient, err := createMesosClient(md, httpClient, httpTransport, cacheTTL)

	if err != nil {
		t.Fatalf("createMesosClient does not yield an error")
	}

	slaveNodes, err := mesosClient.listSlaves(context.TODO())

	if err != nil {
		t.Fatalf("listSlaves does not yield an error")
	}
	if len(slaveNodes) != 3 {
		t.Fatalf("listSlaves yields a collection of size 3")
	}

	expectedHostnames := map[string]struct{}{
		"mesos1.internal.example.org.fail": {},
		"mesos2.internal.example.org.fail": {},
		"mesos3.internal.example.org.fail": {},
	}

	actualHostnames := make(map[string]struct{})
	for _, node := range slaveNodes {
		actualHostnames[node.hostname] = struct{}{}
	}

	if !reflect.DeepEqual(expectedHostnames, actualHostnames) {
		t.Fatalf("listSlaves yields a collection with the expected hostnames")
	}
}

// test mesos.clusterName
func Test_clusterName(t *testing.T) {
	defer log.Flush()
	md := FakeMasterDetector{}
	httpServer, httpClient, httpTransport := makeHttpMocks()
	defer httpServer.Close()
	cacheTTL := 500 * time.Millisecond
	mesosClient, err := createMesosClient(md, httpClient, httpTransport, cacheTTL)

	name, err := mesosClient.clusterName(context.TODO())

	if err != nil {
		t.Fatalf("clusterName does not yield an error")
	}
	if name != defaultClusterName {
		t.Fatalf("clusterName yields the expected (default) value")
	}
}
