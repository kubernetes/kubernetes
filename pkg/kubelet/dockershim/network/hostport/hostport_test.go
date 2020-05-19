// +build !dockerless

/*
Copyright 2017 The Kubernetes Authors.

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

package hostport

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
)

type fakeSocket struct {
	port     int32
	protocol string
	closed   bool
}

func (f *fakeSocket) Close() error {
	if f.closed {
		return fmt.Errorf("Socket %q.%s already closed!", f.port, f.protocol)
	}
	f.closed = true
	return nil
}

func NewFakeSocketManager() *fakeSocketManager {
	return &fakeSocketManager{mem: make(map[hostport]*fakeSocket)}
}

type fakeSocketManager struct {
	mem map[hostport]*fakeSocket
}

func (f *fakeSocketManager) openFakeSocket(hp *hostport) (closeable, error) {
	if socket, ok := f.mem[*hp]; ok && !socket.closed {
		return nil, fmt.Errorf("hostport is occupied")
	}
	fs := &fakeSocket{hp.port, hp.protocol, false}
	f.mem[*hp] = fs
	return fs, nil
}

func TestEnsureKubeHostportChains(t *testing.T) {
	interfaceName := "cbr0"
	builtinChains := []string{"PREROUTING", "OUTPUT"}
	jumpRule := "-m comment --comment \"kube hostport portals\" -m addrtype --dst-type LOCAL -j KUBE-HOSTPORTS"
	masqRule := "-m comment --comment \"SNAT for localhost access to hostports\" -o cbr0 -s 127.0.0.0/8 -j MASQUERADE"

	fakeIPTables := NewFakeIPTables()
	assert.NoError(t, ensureKubeHostportChains(fakeIPTables, interfaceName))

	_, _, err := fakeIPTables.getChain(utiliptables.TableNAT, utiliptables.Chain("KUBE-HOSTPORTS"))
	assert.NoError(t, err)

	_, chain, err := fakeIPTables.getChain(utiliptables.TableNAT, utiliptables.ChainPostrouting)
	assert.NoError(t, err)
	assert.EqualValues(t, len(chain.rules), 1)
	assert.Contains(t, chain.rules[0], masqRule)

	for _, chainName := range builtinChains {
		_, chain, err := fakeIPTables.getChain(utiliptables.TableNAT, utiliptables.Chain(chainName))
		assert.NoError(t, err)
		assert.EqualValues(t, len(chain.rules), 1)
		assert.Contains(t, chain.rules[0], jumpRule)
	}

}
