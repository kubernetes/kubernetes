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
	"fmt"
	"net"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
)

type fakeChain struct {
	name  utiliptables.Chain
	rules []string
}

type fakeTable struct {
	name   utiliptables.Table
	chains map[string]*fakeChain
}

type fakeIPTables struct {
	tables        map[string]*fakeTable
	builtinChains map[string]sets.String
	protocol      utiliptables.Protocol
}

func NewFakeIPTables() *fakeIPTables {
	return &fakeIPTables{
		tables: make(map[string]*fakeTable),
		builtinChains: map[string]sets.String{
			string(utiliptables.TableFilter): sets.NewString("INPUT", "FORWARD", "OUTPUT"),
			string(utiliptables.TableNAT):    sets.NewString("PREROUTING", "INPUT", "OUTPUT", "POSTROUTING"),
			string(utiliptables.TableMangle): sets.NewString("PREROUTING", "INPUT", "FORWARD", "OUTPUT", "POSTROUTING"),
		},
		protocol: utiliptables.ProtocolIPv4,
	}
}

func (f *fakeIPTables) getTable(tableName utiliptables.Table) (*fakeTable, error) {
	table, ok := f.tables[string(tableName)]
	if !ok {
		return nil, fmt.Errorf("table %s does not exist", tableName)
	}
	return table, nil
}

func (f *fakeIPTables) getChain(tableName utiliptables.Table, chainName utiliptables.Chain) (*fakeTable, *fakeChain, error) {
	table, err := f.getTable(tableName)
	if err != nil {
		return nil, nil, err
	}

	chain, ok := table.chains[string(chainName)]
	if !ok {
		return table, nil, fmt.Errorf("chain %s/%s does not exist", tableName, chainName)
	}

	return table, chain, nil
}

func (f *fakeIPTables) ensureChain(tableName utiliptables.Table, chainName utiliptables.Chain) (bool, *fakeChain) {
	table, chain, err := f.getChain(tableName, chainName)
	if err != nil {
		// either table or table+chain don't exist yet
		if table == nil {
			table = &fakeTable{
				name:   tableName,
				chains: make(map[string]*fakeChain),
			}
			f.tables[string(tableName)] = table
		}
		chain := &fakeChain{
			name:  chainName,
			rules: make([]string, 0),
		}
		table.chains[string(chainName)] = chain
		return false, chain
	}
	return true, chain
}

func (f *fakeIPTables) EnsureChain(tableName utiliptables.Table, chainName utiliptables.Chain) (bool, error) {
	existed, _ := f.ensureChain(tableName, chainName)
	return existed, nil
}

func (f *fakeIPTables) FlushChain(tableName utiliptables.Table, chainName utiliptables.Chain) error {
	_, chain, err := f.getChain(tableName, chainName)
	if err != nil {
		return err
	}
	chain.rules = make([]string, 0)
	return nil
}

func (f *fakeIPTables) DeleteChain(tableName utiliptables.Table, chainName utiliptables.Chain) error {
	table, _, err := f.getChain(tableName, chainName)
	if err != nil {
		return err
	}
	delete(table.chains, string(chainName))
	return nil
}

// Returns index of rule in array; < 0 if rule is not found
func findRule(chain *fakeChain, rule string) int {
	for i, candidate := range chain.rules {
		if rule == candidate {
			return i
		}
	}
	return -1
}

func (f *fakeIPTables) ensureRule(position utiliptables.RulePosition, tableName utiliptables.Table, chainName utiliptables.Chain, rule string) (bool, error) {
	_, chain, err := f.getChain(tableName, chainName)
	if err != nil {
		_, chain = f.ensureChain(tableName, chainName)
	}

	rule, err = normalizeRule(rule)
	if err != nil {
		return false, err
	}
	ruleIdx := findRule(chain, rule)
	if ruleIdx >= 0 {
		return true, nil
	}

	if position == utiliptables.Prepend {
		chain.rules = append([]string{rule}, chain.rules...)
	} else if position == utiliptables.Append {
		chain.rules = append(chain.rules, rule)
	} else {
		return false, fmt.Errorf("unknown position argument %q", position)
	}

	return false, nil
}

func normalizeRule(rule string) (string, error) {
	normalized := ""
	remaining := strings.TrimSpace(rule)
	for {
		var end int

		if strings.HasPrefix(remaining, "--to-destination=") {
			remaining = strings.Replace(remaining, "=", " ", 1)
		}

		if remaining[0] == '"' {
			end = strings.Index(remaining[1:], "\"")
			if end < 0 {
				return "", fmt.Errorf("invalid rule syntax: mismatched quotes")
			}
			end += 2
		} else {
			end = strings.Index(remaining, " ")
			if end < 0 {
				end = len(remaining)
			}
		}
		arg := remaining[:end]

		// Normalize un-prefixed IP addresses like iptables does
		if net.ParseIP(arg) != nil {
			arg = arg + "/32"
		}

		if len(normalized) > 0 {
			normalized += " "
		}
		normalized += strings.TrimSpace(arg)
		if len(remaining) == end {
			break
		}
		remaining = remaining[end+1:]
	}
	return normalized, nil
}

func (f *fakeIPTables) EnsureRule(position utiliptables.RulePosition, tableName utiliptables.Table, chainName utiliptables.Chain, args ...string) (bool, error) {
	ruleArgs := make([]string, 0)
	for _, arg := range args {
		// quote args with internal spaces (like comments)
		if strings.Contains(arg, " ") {
			arg = fmt.Sprintf("\"%s\"", arg)
		}
		ruleArgs = append(ruleArgs, arg)
	}
	return f.ensureRule(position, tableName, chainName, strings.Join(ruleArgs, " "))
}

func (f *fakeIPTables) DeleteRule(tableName utiliptables.Table, chainName utiliptables.Chain, args ...string) error {
	_, chain, err := f.getChain(tableName, chainName)
	if err == nil {
		rule := strings.Join(args, " ")
		ruleIdx := findRule(chain, rule)
		if ruleIdx < 0 {
			return nil
		}
		chain.rules = append(chain.rules[:ruleIdx], chain.rules[ruleIdx+1:]...)
	}
	return nil
}

func (f *fakeIPTables) IsIPv6() bool {
	return f.protocol == utiliptables.ProtocolIPv6
}

func (f *fakeIPTables) Protocol() utiliptables.Protocol {
	return f.protocol
}

func saveChain(chain *fakeChain, data *bytes.Buffer) {
	for _, rule := range chain.rules {
		data.WriteString(fmt.Sprintf("-A %s %s\n", chain.name, rule))
	}
}

func (f *fakeIPTables) SaveInto(tableName utiliptables.Table, buffer *bytes.Buffer) error {
	table, err := f.getTable(tableName)
	if err != nil {
		return err
	}

	buffer.WriteString(fmt.Sprintf("*%s\n", table.name))

	rules := bytes.NewBuffer(nil)
	for _, chain := range table.chains {
		buffer.WriteString(fmt.Sprintf(":%s - [0:0]\n", string(chain.name)))
		saveChain(chain, rules)
	}
	buffer.Write(rules.Bytes())
	buffer.WriteString("COMMIT\n")
	return nil
}

func (f *fakeIPTables) restore(restoreTableName utiliptables.Table, data []byte, flush utiliptables.FlushFlag) error {
	allLines := string(data)
	buf := bytes.NewBuffer(data)
	var tableName utiliptables.Table
	for {
		line, err := buf.ReadString('\n')
		if err != nil {
			break
		}
		if line[0] == '#' {
			continue
		}

		line = strings.TrimSuffix(line, "\n")
		if strings.HasPrefix(line, "*") {
			tableName = utiliptables.Table(line[1:])
		}
		if tableName != "" {
			if restoreTableName != "" && restoreTableName != tableName {
				continue
			}
			if strings.HasPrefix(line, ":") {
				chainName := utiliptables.Chain(strings.Split(line[1:], " ")[0])
				if flush == utiliptables.FlushTables {
					table, chain, _ := f.getChain(tableName, chainName)
					if chain != nil {
						delete(table.chains, string(chainName))
					}
				}
				_, _ = f.ensureChain(tableName, chainName)
				// The --noflush option for iptables-restore doesn't work for user-defined chains, only builtin chains.
				// We should flush user-defined chains if the chain is not to be deleted
				if !f.isBuiltinChain(tableName, chainName) && !strings.Contains(allLines, "-X "+string(chainName)) {
					if err := f.FlushChain(tableName, chainName); err != nil {
						return err
					}
				}
			} else if strings.HasPrefix(line, "-A") {
				parts := strings.Split(line, " ")
				if len(parts) < 3 {
					return fmt.Errorf("invalid iptables rule '%s'", line)
				}
				chainName := utiliptables.Chain(parts[1])
				rule := strings.TrimPrefix(line, fmt.Sprintf("-A %s ", chainName))
				_, err := f.ensureRule(utiliptables.Append, tableName, chainName, rule)
				if err != nil {
					return err
				}
			} else if strings.HasPrefix(line, "-I") {
				parts := strings.Split(line, " ")
				if len(parts) < 3 {
					return fmt.Errorf("invalid iptables rule '%s'", line)
				}
				chainName := utiliptables.Chain(parts[1])
				rule := strings.TrimPrefix(line, fmt.Sprintf("-I %s ", chainName))
				_, err := f.ensureRule(utiliptables.Prepend, tableName, chainName, rule)
				if err != nil {
					return err
				}
			} else if strings.HasPrefix(line, "-X") {
				parts := strings.Split(line, " ")
				if len(parts) < 2 {
					return fmt.Errorf("invalid iptables rule '%s'", line)
				}
				if err := f.DeleteChain(tableName, utiliptables.Chain(parts[1])); err != nil {
					return err
				}
			} else if line == "COMMIT" {
				if restoreTableName == tableName {
					return nil
				}
				tableName = ""
			}
		}
	}

	return nil
}

func (f *fakeIPTables) Restore(tableName utiliptables.Table, data []byte, flush utiliptables.FlushFlag, counters utiliptables.RestoreCountersFlag) error {
	return f.restore(tableName, data, flush)
}

func (f *fakeIPTables) RestoreAll(data []byte, flush utiliptables.FlushFlag, counters utiliptables.RestoreCountersFlag) error {
	return f.restore("", data, flush)
}

func (f *fakeIPTables) Monitor(canary utiliptables.Chain, tables []utiliptables.Table, reloadFunc func(), interval time.Duration, stopCh <-chan struct{}) {
}

func (f *fakeIPTables) isBuiltinChain(tableName utiliptables.Table, chainName utiliptables.Chain) bool {
	if builtinChains, ok := f.builtinChains[string(tableName)]; ok && builtinChains.Has(string(chainName)) {
		return true
	}
	return false
}

func (f *fakeIPTables) HasRandomFully() bool {
	return false
}
