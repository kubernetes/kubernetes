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

package e2e

import (
	"fmt"
	"io/ioutil"
	"net"
	"strings"
	"sync"
)

type command struct {
	cmd       string
	component string
}

func CoreDump(dir string) {
	c, err := loadClient()
	if err != nil {
		fmt.Printf("Error creating client: %v", err)
		return
	}
	provider := testContext.Provider

	// requires ssh
	if !providerIs("gce", "gke") {
		fmt.Printf("Skipping SSH core dump, which is not implemented for %s", provider)
		return
	}

	// Get all nodes' external IPs.
	hosts, err := NodeSSHHosts(c)
	if err != nil {
		fmt.Printf("Error getting node hostnames: %v", err)
		return
	}

	cmds := []command{
		{"cat /var/log/kubelet.log", "kubelet"},
		{"cat /var/log/kube-proxy.log", "kube-proxy"},
		{"cat /var/log/monit.log", "monit"},
	}
	logCore(cmds, hosts, dir, provider)

	// I wish there was a better way to get the master IP...
	config, err := loadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v", err)
	}
	ix := strings.LastIndex(config.Host, "/")
	master := net.JoinHostPort(config.Host[ix+1:], "22")
	cmds = []command{
		{"cat /var/log/kubelet.log", "kubelet"},
		{"cat /var/log/kube-apiserver.log", "kube-apiserver"},
		{"cat /var/log/kube-scheduler.log", "kube-scheduler"},
		{"cat /var/log/kube-controller-manager.log", "kube-controller-manager"},
		{"cat /var/log/monit.log", "monit"},
	}
	logCore(cmds, []string{master}, dir, provider)
}

func logCore(cmds []command, hosts []string, dir, provider string) {
	wg := &sync.WaitGroup{}
	// Run commands on all nodes via SSH.
	for _, cmd := range cmds {
		fmt.Printf("SSH'ing to all nodes and running %s\n", cmd.cmd)
		for _, host := range hosts {
			wg.Add(1)
			go func(cmd command, host string) {
				defer wg.Done()
				logfile := fmt.Sprintf("%s/%s-%s.log", dir, host, cmd.component)
				fmt.Printf("Writing to %s.\n", logfile)
				stdout, stderr, _, err := SSH(cmd.cmd, host, provider)
				if err != nil {
					fmt.Printf("Error running command: %v\n", err)
				}
				if err := ioutil.WriteFile(logfile, []byte(stdout+stderr), 0777); err != nil {
					fmt.Printf("Error writing logfile: %v\n", err)
				}
			}(cmd, host)
		}
	}
	wg.Wait()
}
