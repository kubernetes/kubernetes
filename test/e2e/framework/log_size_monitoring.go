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

package framework

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	clientset "k8s.io/client-go/kubernetes"

	// TODO: Remove the following imports (ref: https://github.com/kubernetes/kubernetes/issues/81245)
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
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
	client      clientset.Interface
	stopChannel chan bool
	// data stores LogSizeData groupped per IP and log_path
	data          *LogsSizeData
	masterAddress string
	nodeAddresses []string
	wg            sync.WaitGroup
	workChannel   chan WorkItem
	workers       []*LogSizeGatherer
}

// SingleLogSummary is a structure for handling average generation rate and number of probes.
type SingleLogSummary struct {
	AverageGenerationRate int
	NumberOfProbes        int
}

// LogSizeDataTimeseries is map of timestamped size.
type LogSizeDataTimeseries map[string]map[string][]TimestampedSize

// LogsSizeDataSummary is map of log summary.
// node -> file -> data
type LogsSizeDataSummary map[string]map[string]SingleLogSummary

// PrintHumanReadable returns string of log size data summary.
// TODO: make sure that we don't need locking here
func (s *LogsSizeDataSummary) PrintHumanReadable() string {
	buf := &bytes.Buffer{}
	w := tabwriter.NewWriter(buf, 1, 0, 1, ' ', 0)
	fmt.Fprintf(w, "host\tlog_file\taverage_rate (B/s)\tnumber_of_probes\n")
	for k, v := range *s {
		fmt.Fprintf(w, "%v\t\t\t\n", k)
		for path, data := range v {
			fmt.Fprintf(w, "\t%v\t%v\t%v\n", path, data.AverageGenerationRate, data.NumberOfProbes)
		}
	}
	w.Flush()
	return buf.String()
}

// PrintJSON returns the summary of log size data with JSON format.
func (s *LogsSizeDataSummary) PrintJSON() string {
	return PrettyPrintJSON(*s)
}

// SummaryKind returns the summary of log size data summary.
func (s *LogsSizeDataSummary) SummaryKind() string {
	return "LogSizeSummary"
}

// LogsSizeData is a structure for handling timeseries of log size data and lock.
type LogsSizeData struct {
	data LogSizeDataTimeseries
	lock sync.Mutex
}

// WorkItem is a command for a worker that contains an IP of machine from which we want to
// gather data and paths to all files we're interested in.
type WorkItem struct {
	ip                string
	paths             []string
	backoffMultiplier int
}

func prepareData(masterAddress string, nodeAddresses []string) *LogsSizeData {
	data := make(LogSizeDataTimeseries)
	ips := append(nodeAddresses, masterAddress)
	for _, ip := range ips {
		data[ip] = make(map[string][]TimestampedSize)
	}
	return &LogsSizeData{
		data: data,
		lock: sync.Mutex{},
	}
}

func (d *LogsSizeData) addNewData(ip, path string, timestamp time.Time, size int) {
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

// NewLogsVerifier creates a new LogsSizeVerifier which will stop when stopChannel is closed
func NewLogsVerifier(c clientset.Interface, stopChannel chan bool) *LogsSizeVerifier {
	nodeAddresses, err := e2essh.NodeSSHHosts(c)
	ExpectNoError(err)
	instanceAddress := APIAddress() + ":22"

	workChannel := make(chan WorkItem, len(nodeAddresses)+1)
	workers := make([]*LogSizeGatherer, workersNo)

	verifier := &LogsSizeVerifier{
		client:        c,
		stopChannel:   stopChannel,
		data:          prepareData(instanceAddress, nodeAddresses),
		masterAddress: instanceAddress,
		nodeAddresses: nodeAddresses,
		wg:            sync.WaitGroup{},
		workChannel:   workChannel,
		workers:       workers,
	}
	verifier.wg.Add(workersNo)
	for i := 0; i < workersNo; i++ {
		workers[i] = &LogSizeGatherer{
			stopChannel: stopChannel,
			data:        verifier.data,
			wg:          &verifier.wg,
			workChannel: workChannel,
		}
	}
	return verifier
}

// GetSummary returns a summary (average generation rate and number of probes) of the data gathered by LogSizeVerifier
func (s *LogsSizeVerifier) GetSummary() *LogsSizeDataSummary {
	result := make(LogsSizeDataSummary)
	for k, v := range s.data.data {
		result[k] = make(map[string]SingleLogSummary)
		for path, data := range v {
			if len(data) > 1 {
				last := data[len(data)-1]
				first := data[0]
				rate := (last.size - first.size) / int(last.timestamp.Sub(first.timestamp)/time.Second)
				result[k][path] = SingleLogSummary{
					AverageGenerationRate: rate,
					NumberOfProbes:        len(data),
				}
			}
		}
	}
	return &result
}

// Run starts log size gathering. It starts a gorouting for every worker and then blocks until stopChannel is closed
func (s *LogsSizeVerifier) Run() {
	s.workChannel <- WorkItem{
		ip:                s.masterAddress,
		paths:             masterLogsToCheck,
		backoffMultiplier: 1,
	}
	for _, node := range s.nodeAddresses {
		s.workChannel <- WorkItem{
			ip:                node,
			paths:             nodeLogsToCheck,
			backoffMultiplier: 1,
		}
	}
	for _, worker := range s.workers {
		go worker.Run()
	}
	<-s.stopChannel
	s.wg.Wait()
}

// Run starts log size gathering.
func (g *LogSizeGatherer) Run() {
	for g.Work() {
	}
}

func (g *LogSizeGatherer) pushWorkItem(workItem WorkItem) {
	select {
	case <-time.After(time.Duration(workItem.backoffMultiplier) * pollingPeriod):
		g.workChannel <- workItem
	case <-g.stopChannel:
		return
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
	sshResult, err := e2essh.SSH(
		fmt.Sprintf("ls -l %v | awk '{print $9, $5}' | tr '\n' ' '", strings.Join(workItem.paths, " ")),
		workItem.ip,
		TestContext.Provider,
	)
	if err != nil {
		Logf("Error while trying to SSH to %v, skipping probe. Error: %v", workItem.ip, err)
		// In case of repeated error give up.
		if workItem.backoffMultiplier >= 128 {
			Logf("Failed to ssh to a node %v multiple times in a row. Giving up.", workItem.ip)
			g.wg.Done()
			return false
		}
		workItem.backoffMultiplier *= 2
		go g.pushWorkItem(workItem)
		return true
	}
	workItem.backoffMultiplier = 1
	results := strings.Split(sshResult.Stdout, " ")

	now := time.Now()
	for i := 0; i+1 < len(results); i = i + 2 {
		path := results[i]
		size, err := strconv.Atoi(results[i+1])
		if err != nil {
			Logf("Error during conversion to int: %v, skipping data. Error: %v", results[i+1], err)
			continue
		}
		g.data.addNewData(workItem.ip, path, now, size)
	}
	go g.pushWorkItem(workItem)
	return true
}
