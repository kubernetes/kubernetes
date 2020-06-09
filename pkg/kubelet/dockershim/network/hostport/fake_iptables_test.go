// +build !dockerless

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

package hostport

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
)

func TestRestoreFlushRules(t *testing.T) {
	iptables := NewFakeIPTables()
	rules := [][]string{
		{"-A", "KUBE-HOSTPORTS", "-m comment --comment \"pod3_ns1 hostport 8443\" -m tcp -p tcp --dport 8443 -j KUBE-HP-5N7UH5JAXCVP5UJR"},
		{"-A", "POSTROUTING", "-m comment --comment \"SNAT for localhost access to hostports\" -o cbr0 -s 127.0.0.0/8 -j MASQUERADE"},
	}
	natRules := bytes.NewBuffer(nil)
	writeLine(natRules, "*nat")
	for _, rule := range rules {
		_, err := iptables.EnsureChain(utiliptables.TableNAT, utiliptables.Chain(rule[1]))
		assert.NoError(t, err)
		_, err = iptables.ensureRule(utiliptables.RulePosition(rule[0]), utiliptables.TableNAT, utiliptables.Chain(rule[1]), rule[2])
		assert.NoError(t, err)

		writeLine(natRules, utiliptables.MakeChainLine(utiliptables.Chain(rule[1])))
	}
	writeLine(natRules, "COMMIT")
	assert.NoError(t, iptables.Restore(utiliptables.TableNAT, natRules.Bytes(), utiliptables.NoFlushTables, utiliptables.RestoreCounters))
	natTable, ok := iptables.tables[string(utiliptables.TableNAT)]
	assert.True(t, ok)
	// check KUBE-HOSTPORTS chain, should have been cleaned up
	hostportChain, ok := natTable.chains["KUBE-HOSTPORTS"]
	assert.True(t, ok)
	assert.Equal(t, 0, len(hostportChain.rules))

	// check builtin chains, should not been cleaned up
	postroutingChain, ok := natTable.chains["POSTROUTING"]
	assert.True(t, ok, string(postroutingChain.name))
	assert.Equal(t, 1, len(postroutingChain.rules))
}
