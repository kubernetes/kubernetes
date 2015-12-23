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
	"bytes"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	client "k8s.io/kubernetes/pkg/client/unversioned"
)

const (
	// Minimal period between polling log sizes from components
	pollingPeriod            = 60 * time.Second
	workersNo                = 5
	kubeletLogsPath          = "/var/log/kubelet.log"
	kubeProxyLogsPath        = "/var/log/kube-proxy.log"
	kubeAddonsLogsPath       = "/var/log/kube-addons.log"
	kubeMasterAddonsLogsPath = "/var/log/kube-master-addons.log"
	apiServerLogsPath        = "/var/log/kube-apiserver.log"
	controllersLogsPath      = "/var/log/kube-controller-manager.log"
	schedulerLogsPath        = "/var/log/kube-scheduler.log"
)

var (
	nodeLogsToCheck   = []string{kubeletLogsPath, kubeProxyLogsPath}
	masterLogsToCheck = []string{kubeletLogsPath, kubeAddonsLogsPath, kubeMasterAddonsLogsPath,
		apiServerLogsPath, controllersLogsPath, schedulerLogsPath}
)

// TimestampedSize contains a size together with a time of measurement.
type TimestampedSize struct {
	timestamp time.Time
	size      int
}

// LogSizeGatherer is a worker which grabs a WorkItem from the channel and does assigned work.
type LogSizeGatherer struct {
	stopChannel chan bool
	data        *LogsSizeData
	wg          *sync.WaitGroup
	workChannel chan WorkItem
}

// LogsSizeVerifier gathers data about log files sizes from master and node machines.
// It oversees a <workersNo> workers which do the gathering.
type LogsSizeVerifier struct {
	client      *client.Client
	stopChannel chan bool
	// data stores LogSizeData groupped per IP and log_path
	data          LogsSizeData
	masterAddress string
	nodeAddresses []string
	wg            sync.WaitGroup
	workChannel   chan WorkItem
	workers       []*LogSizeGatherer
}

type LogsSizeData struct {
	data map[string]map[string][]TimestampedSize
	lock sync.Mutex
}

// WorkItem is a command for a worker that contains an IP of machine from which we want to
// gather data and paths to all files we're interested in.
type WorkItem struct {
	ip    string
	paths []string
}

func prepareData(masterAddress string, nodeAddresses []string) LogsSizeData {
	data := make(map[string]map[string][]TimestampedSize)
	ips := append(nodeAddresses, masterAddress)
	for _, ip := range ips {
		data[ip] = make(map[string][]TimestampedSize)
	}
	return LogsSizeData{
		data: data,
		lock: sync.Mutex{},
	}
}

func (d *LogsSizeData) AddNewData(ip, path string, timestamp time.Time, size int) {
	d.lock.Lock()
	defer d.lock.Unlock()
	d.data[ip][path] = append(
		d.data[ip][path],
		TimestampedSize{
			timestamp: timestamp,
			size:      size,
		},
	)
}

func (d *LogsSizeData) PrintData() string {
	d.lock.Lock()
	defer d.lock.Unlock()
	buf := &bytes.Buffer{}
	w := tabwriter.NewWriter(buf, 1, 0, 1, ' ', 0)
	fmt.Fprintf(w, "host\tlog_file\taverage_rate (B/s)\tnumber_of_probes\n")
	for k, v := range d.data {
		fmt.Fprintf(w, "%v\t\t\t\n", k)
		for path, data := range v {
			if len(data) > 1 {
				last := data[len(data)-1]
				first := data[0]
				rate := (last.size - first.size) / int(last.timestamp.Sub(first.timestamp)/time.Second)
				fmt.Fprintf(w, "\t%v\t%v\t%v\n", path, rate, len(data))
			}
		}
	}
	w.Flush()
	return buf.String()
}

// NewLogsVerifier creates a new LogsSizeVerifier which will stop when stopChannel is closed
func NewLogsVerifier(c *client.Client, stopChannel chan bool) *LogsSizeVerifier {
	nodeAddresses, err := NodeSSHHosts(c)
	expectNoError(err)
	masterAddress := getMasterHost() + ":22"

	workChannel := make(chan WorkItem, len(nodeAddresses)+1)
	workers := make([]*LogSizeGatherer, workersNo)

	verifier := &LogsSizeVerifier{
		client:        c,
		stopChannel:   stopChannel,
		data:          prepareData(masterAddress, nodeAddresses),
		masterAddress: masterAddress,
		nodeAddresses: nodeAddresses,
		wg:            sync.WaitGroup{},
		workChannel:   workChannel,
		workers:       workers,
	}
	verifier.wg.Add(workersNo)
	for i := 0; i < workersNo; i++ {
		workers[i] = &LogSizeGatherer{
			stopChannel: stopChannel,
			data:        &verifier.data,
			wg:          &verifier.wg,
			workChannel: workChannel,
		}
	}
	return verifier
}

// PrintData returns a string with formated results
func (v *LogsSizeVerifier) PrintData() string {
	return v.data.PrintData()
}

// Run starts log size gathering. It starts a gorouting for every worker and then blocks until stopChannel is closed
func (v *LogsSizeVerifier) Run() {
	v.workChannel <- WorkItem{
		ip:    v.masterAddress,
		paths: masterLogsToCheck,
	}
	for _, node := range v.nodeAddresses {
		v.workChannel <- WorkItem{
			ip:    node,
			paths: nodeLogsToCheck,
		}
	}
	for _, worker := range v.workers {
		go worker.Run()
	}
	<-v.stopChannel
	v.wg.Wait()

	Logf("\n%v", v.PrintData())
}

func (g *LogSizeGatherer) Run() {
	for g.Work() {
	}
}

// Work does a single unit of work: tries to take out a WorkItem from the queue, ssh-es into a given machine,
// gathers data, writes it to the shared <data> map, and creates a gorouting which reinserts work item into
// the queue with a <pollingPeriod> delay. Returns false if worker should exit.
func (g *LogSizeGatherer) Work() bool {
	var workItem WorkItem
	select {
	case <-g.stopChannel:
		g.wg.Done()
		return false
	case workItem = <-g.workChannel:
	}
	sshResult, err := SSH(
		fmt.Sprintf("ls -l %v | awk '{print $9, $5}' | tr '\n' ' '", strings.Join(workItem.paths, " ")),
		workItem.ip,
		testContext.Provider,
	)
	if err != nil {
		Logf("Error while trying to SSH to %v, skipping probe. Error: %v", workItem.ip, err)
		g.workChannel <- workItem
		return true
	}
	results := strings.Split(sshResult.Stdout, " ")

	now := time.Now()
	for i := 0; i+1 < len(results); i = i + 2 {
		path := results[i]
		size, err := strconv.Atoi(results[i+1])
		if err != nil {
			Logf("Error during conversion to int: %v, skipping data. Error: %v", results[i+1], err)
			continue
		}
		g.data.AddNewData(workItem.ip, path, now, size)
	}
	go func() {
		time.Sleep(pollingPeriod)
		g.workChannel <- workItem
	}()
	return true
}
