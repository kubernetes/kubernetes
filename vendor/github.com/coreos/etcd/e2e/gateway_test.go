// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package e2e

import (
	"os"
	"strings"
	"testing"

	"github.com/coreos/etcd/pkg/expect"
)

var (
	defaultGatewayEndpoint = "127.0.0.1:23790"
)

func TestGateway(t *testing.T) {
	ec, err := newEtcdProcessCluster(&configNoTLS)
	if err != nil {
		t.Fatal(err)
	}
	defer ec.StopAll()

	eps := strings.Join(ec.grpcEndpoints(), ",")

	p := startGateway(t, eps)
	defer p.Stop()

	os.Setenv("ETCDCTL_API", "3")
	defer os.Unsetenv("ETCDCTL_API")

	err = spawnWithExpect([]string{ctlBinPath, "--endpoints=" + defaultGatewayEndpoint, "put", "foo", "bar"}, "OK\r\n")
	if err != nil {
		t.Errorf("failed to finish put request through gateway: %v", err)
	}
}

func startGateway(t *testing.T, endpoints string) *expect.ExpectProcess {
	p, err := expect.NewExpect(binPath, "gateway", "--endpoints="+endpoints, "start")
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Expect("tcpproxy: ready to proxy client requests to")
	if err != nil {
		t.Fatal(err)
	}
	return p
}
