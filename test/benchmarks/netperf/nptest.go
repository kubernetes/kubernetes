/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

/*
 nptest.go

 Dual-mode program - runs as both the orchestrator and as the worker nodes depending on command line flags
 The RPC API is contained wholly within this file.
*/

package main

// Imports only base Golang packages
import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/rpc"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"sync"
	"time"
)

type Point struct {
	mss       int
	bandwidth string
	index     int
}

var mode string
var port string
var host string
var worker string
var kubenode string
var podname string

var workerStateMap map[string]*workerState

var iperf_tcp_output_regexp *regexp.Regexp
var iperf_udp_output_regexp *regexp.Regexp
var netperf_output_regexp *regexp.Regexp

var dataPoints map[string][]Point
var dataPointKeys []string
var datapointsFlushed bool

var global_lock sync.Mutex

const (
	WORKER_MODE         = "worker"
	ORCHESTRATOR_MODE   = "orchestrator"
	IPERF3_PATH         = "/usr/bin/iperf3"
	NETPERF_PATH        = "/usr/local/bin/netperf"
	NETPERF_SERVER_PATH = "/usr/local/bin/netserver"
	OUTPUT_CAPTURE_FILE = "/tmp/output.txt"
	CSVFILE             = "/tmp/netperf.csv"
	MSS_MIN             = 96
	MSS_MAX             = 1460
	MSS_STEP_SIZE       = 256
	PARALLELSTREAMS     = "8"
)

const (
	IPERF_TCP_TEST = iota
	IPERF_UDP_TEST = iota
	NETPERF_TEST   = iota
)

type NetPerfRpc int
type ClientRegistrationData struct {
	Host     string
	KubeNode string
	Worker   string
	IP       string
}

type IperfClientWorkItem struct {
	Host string
	Port string
	MSS  int
	Type int
}

type IperfServerWorkItem struct {
	ListenPort string
	Timeout    int
}

type WorkItem struct {
	IsClientItem bool
	IsServerItem bool
	IsIdle       bool
	ClientItem   IperfClientWorkItem
	ServerItem   IperfServerWorkItem
}

type workerState struct {
	sentServerItem bool
	idle           bool
	IP             string
	worker         string
}

type WorkerOutput struct {
	Output string
	Code   int
	Worker string
	Type   int
}

type Testcase struct {
	SourceNode      string
	DestinationNode string
	Label           string
	ClusterIP       bool
	Finished        bool
	MSS             int
	Type            int
}

var testcases []*Testcase
var currentJobIndex int

func init() {
	flag.StringVar(&mode, "mode", "worker", "Mode for the daemon (worker | orchestrator)")
	flag.StringVar(&port, "port", "5202", "Port to listen on (defaults to 5202)")
	flag.StringVar(&host, "host", "", "IP address to bind to (defaults to 0.0.0.0)")

	workerStateMap = make(map[string]*workerState)
	testcases = []*Testcase{
		{SourceNode: "netperf-w1", DestinationNode: "netperf-w2", Label: "1 iperf TCP. Same VM using Pod IP", Type: IPERF_TCP_TEST, ClusterIP: false, MSS: MSS_MIN},
		{SourceNode: "netperf-w1", DestinationNode: "netperf-w2", Label: "2 iperf TCP. Same VM using Virtual IP", Type: IPERF_TCP_TEST, ClusterIP: true, MSS: MSS_MIN},
		{SourceNode: "netperf-w1", DestinationNode: "netperf-w3", Label: "3 iperf TCP. Remote VM using Pod IP", Type: IPERF_TCP_TEST, ClusterIP: false, MSS: MSS_MIN},
		{SourceNode: "netperf-w3", DestinationNode: "netperf-w2", Label: "4 iperf TCP. Remote VM using Virtual IP", Type: IPERF_TCP_TEST, ClusterIP: true, MSS: MSS_MIN},
		{SourceNode: "netperf-w2", DestinationNode: "netperf-w2", Label: "5 iperf TCP. Hairpin Pod to own Virtual IP", Type: IPERF_TCP_TEST, ClusterIP: true, MSS: MSS_MIN},
		{SourceNode: "netperf-w1", DestinationNode: "netperf-w2", Label: "6 iperf UDP. Same VM using Pod IP", Type: IPERF_UDP_TEST, ClusterIP: false, MSS: MSS_MAX},
		{SourceNode: "netperf-w1", DestinationNode: "netperf-w2", Label: "7 iperf UDP. Same VM using Virtual IP", Type: IPERF_UDP_TEST, ClusterIP: true, MSS: MSS_MAX},
		{SourceNode: "netperf-w1", DestinationNode: "netperf-w3", Label: "8 iperf UDP. Remote VM using Pod IP", Type: IPERF_UDP_TEST, ClusterIP: false, MSS: MSS_MAX},
		{SourceNode: "netperf-w3", DestinationNode: "netperf-w2", Label: "9 iperf UDP. Remote VM using Virtual IP", Type: IPERF_UDP_TEST, ClusterIP: true, MSS: MSS_MAX},
		{SourceNode: "netperf-w1", DestinationNode: "netperf-w2", Label: "10 netperf. Same VM using Pod IP", Type: NETPERF_TEST, ClusterIP: false},
		{SourceNode: "netperf-w1", DestinationNode: "netperf-w2", Label: "11 netperf. Same VM using Virtual IP", Type: NETPERF_TEST, ClusterIP: true},
		{SourceNode: "netperf-w1", DestinationNode: "netperf-w3", Label: "12 netperf. Remote VM using Pod IP", Type: NETPERF_TEST, ClusterIP: false},
		{SourceNode: "netperf-w3", DestinationNode: "netperf-w2", Label: "13 netperf. Remote VM using Virtual IP", Type: NETPERF_TEST, ClusterIP: true},
	}

	currentJobIndex = 0

	// Regexes to parse the Mbits/sec out of iperf TCP, UDP and netperf output
	iperf_tcp_output_regexp = regexp.MustCompile("SUM.*\\s+(\\d+)\\sMbits/sec\\s+receiver")
	iperf_udp_output_regexp = regexp.MustCompile("\\s+(\\S+)\\sMbits/sec\\s+\\S+\\s+ms\\s+")
	netperf_output_regexp = regexp.MustCompile("\\s+\\d+\\s+\\d+\\s+\\d+\\s+\\S+\\s+(\\S+)\\s+")

	dataPoints = make(map[string][]Point)
}

func initializeOutputFiles() {
	fd, err := os.OpenFile(OUTPUT_CAPTURE_FILE, os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		fmt.Println("Failed to open output capture file", err)
		os.Exit(2)
	}
	fd.Close()

	fd, err = os.OpenFile(CSVFILE, os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		fmt.Println("Failed to open output capture file", err)
		os.Exit(2)
	}

	fmt.Println("Netperf Test Daemon starting up...")
	fd.Close()
}

func main() {
	initializeOutputFiles()
	flag.Parse()
	if validateParams() == false {
		fmt.Println("Failed to parse cmdline args - fatal error - bailing out")
		os.Exit(1)

	}
	grabEnv()
	fmt.Println("Running as", mode, "...")
	if mode == ORCHESTRATOR_MODE {
		orchestrate()
	} else {
		start_work()
	}
	fmt.Println("Terminating npd")
}

func grabEnv() {
	worker = os.Getenv("worker")
	kubenode = os.Getenv("kubenode")
	podname = os.Getenv("HOSTNAME")
}

func validateParams() (rv bool) {
	rv = true
	if mode != WORKER_MODE && mode != ORCHESTRATOR_MODE {
		fmt.Println("Invalid mode", mode)
		return false
	}

	if len(port) == 0 {
		fmt.Println("Invalid port", port)
		return false
	}

	if (len(host)) == 0 {
		if mode == ORCHESTRATOR_MODE {
			host = os.Getenv("NETPERF_ORCH_SERVICE_HOST")
		} else {
			host = os.Getenv("NETPERF_ORCH_SERVICE_HOST")
		}
	}
	return
}

func allWorkersIdle() bool {
	for _, v := range workerStateMap {
		if v.idle == false {
			return false
		}
	}
	return true
}

func getWorkerPodIP(worker string) string {
	return workerStateMap[worker].IP
}

func allocateWorkToClient(workerS *workerState, reply *WorkItem) {

	if allWorkersIdle() == false {
		reply.IsIdle = true
		return
	}

	// System is all idle - pick up next work item to allocate to client
	for n, v := range testcases {

		if v.Finished == true {
			continue
		}

		if v.SourceNode != workerS.worker {
			reply.IsIdle = true
			return
		}

		_, ok := workerStateMap[v.DestinationNode]
		if ok == false {
			reply.IsIdle = true
			return
		}
		fmt.Println("Requesting jobrun '", v.Label, "' from", v.SourceNode, "to", v.DestinationNode, "for MSS", v.MSS)
		reply.ClientItem.Type = v.Type
		reply.IsClientItem = true
		workerS.idle = false
		currentJobIndex = n

		if v.ClusterIP == false {
			reply.ClientItem.Host = getWorkerPodIP(v.DestinationNode)
		} else {
			reply.ClientItem.Host = os.Getenv("NETPERF_W2_SERVICE_HOST")
		}

		switch {
		case v.Type == IPERF_TCP_TEST || v.Type == IPERF_UDP_TEST:
			reply.ClientItem.Port = "5201"
			reply.ClientItem.MSS = v.MSS

			v.MSS = v.MSS + MSS_STEP_SIZE
			if v.MSS > MSS_MAX {
				v.Finished = true
			}
			return

		case v.Type == NETPERF_TEST:
			reply.ClientItem.Port = "12865"
			return
		}
	}

	for _, v := range testcases {
		if v.Finished == false {
			return
		}
	}

	if datapointsFlushed == false {
		fmt.Println("ALL TESTCASES AND MSS RANGES COMPLETE - WRITING CSV FILE")
		flushDataPointsToCsv()
		datapointsFlushed = true
	}

	reply.IsIdle = true
}

func (t *NetPerfRpc) RegisterClient(data *ClientRegistrationData, reply *WorkItem) error {

	global_lock.Lock()
	defer global_lock.Unlock()

	var workerS *workerState
	workerS, ok := workerStateMap[data.Worker]

	if ok != true {

		// For new clients, trigger an iperf server start immediately
		workerS = &workerState{sentServerItem: true, idle: true, IP: data.IP, worker: data.Worker}
		workerStateMap[data.Worker] = workerS
		reply.IsServerItem = true
		reply.ServerItem.ListenPort = "5201"
		reply.ServerItem.Timeout = 3600
		return nil

	} else {
		// Worker defaults to idle unless the allocateWork routine below assigns an item
		workerS.idle = true
	}

	// Give the worker a new work item or let it idle loop another 5 seconds
	allocateWorkToClient(workerS, reply)
	return nil
}

func writeOutputFile(filename, data string) {

	fd, err := os.OpenFile(filename, os.O_APPEND|os.O_WRONLY, 0666)
	if err != nil {
		fmt.Println("Failed to append to existing file", filename, err)
		return
	}
	defer fd.Close()

	_, err = fd.WriteString(data)
	if err != nil {
		fmt.Println("Failed to append to existing file", filename, err)
	}
}

func registerDataPoint(label string, mss int, value string, index int) {

	sl, ok := dataPoints[label]
	if ok != true {
		dataPoints[label] = []Point{{mss: mss, bandwidth: value, index: index}}
		dataPointKeys = append(dataPointKeys, label)
	} else {
		dataPoints[label] = append(sl, Point{mss: mss, bandwidth: value, index: index})
	}
}

func flushDataPointsToCsv() {

	var buffer string

	// Write the MSS points for the X-axis before dumping all the testcase datapoints
	for _, points := range dataPoints {
		if len(points) == 1 {
			continue
		}
		buffer = fmt.Sprintf("%-45s, Maximum,", "MSS")
		for _, p := range points {
			buffer = buffer + fmt.Sprintf(" %d,", p.mss)
		}
		break
	}
	buffer = buffer + "\n"
	writeOutputFile(CSVFILE, buffer)

	var lines []string
	for _, label := range dataPointKeys {
		buffer = fmt.Sprintf("%-45s,", label)
		points := dataPoints[label]
		var max float64
		for _, p := range points {
			fv, _ := strconv.ParseFloat(p.bandwidth, 64)
			if fv > max {
				max = fv
			}
		}

		buffer = buffer + fmt.Sprintf("%f,", max)
		for _, p := range points {
			buffer = buffer + fmt.Sprintf("%s,", p.bandwidth)
		}
		buffer = buffer + "\n"
		lines = append(lines, buffer)
	}

	for _, l := range lines {
		writeOutputFile(CSVFILE, l)
	}
}

func parseIperfTcpBandwidth(output string) string {
	// Parses the output of iperf3 and grabs the group Mbits/sec from the output
	match := iperf_tcp_output_regexp.FindStringSubmatch(output)
	if match != nil && len(match) > 1 {
		return match[1]
	}
	return "0"
}

func parseIperfUdpBandwidth(output string) string {
	// Parses the output of iperf3 (UDP mode) and grabs the Mbits/sec from the output
	match := iperf_udp_output_regexp.FindStringSubmatch(output)
	if match != nil && len(match) > 1 {
		return match[1]
	}
	return "0"
}

func parseNetperfBandwidth(output string) string {
	// Parses the output of netperf and grabs the Bbits/sec from the output
	match := netperf_output_regexp.FindStringSubmatch(output)
	if match != nil && len(match) > 1 {
		return match[1]
	}
	return "0"
}

func (t *NetPerfRpc) ReceiveOutput(data *WorkerOutput, reply *int) error {
	global_lock.Lock()
	defer global_lock.Unlock()

	testcase := testcases[currentJobIndex]

	var outputLog string
	var bw string

	switch data.Type {
	case IPERF_TCP_TEST:
		mss := testcases[currentJobIndex].MSS - MSS_STEP_SIZE
		outputLog = outputLog + fmt.Sprintln("Received TCP output from worker", data.Worker, "for test", testcase.Label,
			"from", testcase.SourceNode, "to", testcase.DestinationNode, "MSS:", mss, "\n") + data.Output
		go writeOutputFile(OUTPUT_CAPTURE_FILE, outputLog)
		bw = parseIperfTcpBandwidth(data.Output)
		registerDataPoint(testcase.Label, mss, bw, currentJobIndex)

	case IPERF_UDP_TEST:
		mss := testcases[currentJobIndex].MSS - MSS_STEP_SIZE
		outputLog = outputLog + fmt.Sprintln("Received UDP output from worker", data.Worker, "for test", testcase.Label,
			"from", testcase.SourceNode, "to", testcase.DestinationNode, "MSS:", mss, "\n") + data.Output
		go writeOutputFile(OUTPUT_CAPTURE_FILE, outputLog)
		bw = parseIperfUdpBandwidth(data.Output)
		registerDataPoint(testcase.Label, mss, bw, currentJobIndex)

	case NETPERF_TEST:
		outputLog = outputLog + fmt.Sprintln("Received netperf output from worker", data.Worker, "for test", testcase.Label,
			"from", testcase.SourceNode, "to", testcase.DestinationNode, "\n") + data.Output
		go writeOutputFile(OUTPUT_CAPTURE_FILE, outputLog)
		bw = parseNetperfBandwidth(data.Output)
		registerDataPoint(testcase.Label, 0, bw, currentJobIndex)
		testcases[currentJobIndex].Finished = true

	}
	fmt.Println("Jobdone from worker", data.Worker, "Bandwidth was", bw, "Mbits/sec")
	return nil
}

func serveRpcRequests(port string) {
	baseObject := new(NetPerfRpc)
	rpc.Register(baseObject)
	rpc.HandleHTTP()
	listener, e := net.Listen("tcp", ":"+port)
	if e != nil {
		log.Fatal("listen error:", e)
	}
	http.Serve(listener, nil)
}

// Blocking RPC server start - only runs on the orchestrator
func orchestrate() {
	serveRpcRequests("5202")
}

// Walk the list of interfaces and find the first interface that has a valid IP
// Inside a container, there should be only one IP-enabled interface
func getMyIP() string {
	ifaces, err := net.Interfaces()
	if err != nil {
		return "127.0.0.1"
	}

	for _, iface := range ifaces {
		if iface.Flags&net.FlagLoopback == 0 {
			addrs, _ := iface.Addrs()
			for _, addr := range addrs {
				var ip net.IP
				switch v := addr.(type) {
				case *net.IPNet:
					ip = v.IP
				case *net.IPAddr:
					ip = v.IP
				}
				return ip.String()
			}
		}
	}
	return "127.0.0.1"
}

func handle_client_work_item(client *rpc.Client, work_item *WorkItem) {
	fmt.Println("Orchestrator requests worker run item Type:", work_item.ClientItem.Type)
	switch {
	case work_item.ClientItem.Type == IPERF_TCP_TEST || work_item.ClientItem.Type == IPERF_UDP_TEST:
		outputString := iperf_client(work_item.ClientItem.Host, work_item.ClientItem.Port, work_item.ClientItem.MSS, work_item.ClientItem.Type)
		var reply int
		client.Call("NetPerfRpc.ReceiveOutput", WorkerOutput{Output: outputString, Worker: worker, Type: work_item.ClientItem.Type}, &reply)
	case work_item.ClientItem.Type == NETPERF_TEST:
		outputString := netperf_client(work_item.ClientItem.Host, work_item.ClientItem.Port, work_item.ClientItem.Type)
		var reply int
		client.Call("NetPerfRpc.ReceiveOutput", WorkerOutput{Output: outputString, Worker: worker, Type: work_item.ClientItem.Type}, &reply)
	}
	// Client COOLDOWN period before asking for next work item to replenish burst allowance policers etc
	time.Sleep(10 * time.Second)
}

// start_work : Entry point to the worker infinite loop
func start_work() {

	time.Sleep(5 * time.Second)
	for true {
		var timeout time.Duration
		var client *rpc.Client
		var err error

		timeout = 5
		for true {
			fmt.Println("Attempting to connect to orchestrator at", host)
			client, err = rpc.DialHTTP("tcp", host+":"+port)
			if err == nil {
				break
			}
			fmt.Println("RPC connection to ", host, " failed:", err)
			time.Sleep(timeout * time.Second)
		}

		for true {
			client_data := ClientRegistrationData{Host: podname, KubeNode: kubenode, Worker: worker, IP: getMyIP()}
			var work_item WorkItem

			err := client.Call("NetPerfRpc.RegisterClient", client_data, &work_item)
			if err == nil {
				switch {
				case work_item.IsIdle == true:
					time.Sleep(5 * time.Second)
					continue

				case work_item.IsServerItem == true:
					fmt.Println("Orchestrator requests worker run iperf and netperf servers")
					go iperf_server()
					go netperf_server()
					time.Sleep(1 * time.Second)

				case work_item.IsClientItem == true:
					handle_client_work_item(client, &work_item)
				}

			} else {
				// RPC server has probably gone away - attempt to reconnect
				fmt.Println("Error attempting RPC call", err)
				break
			}
		}
	}
}

// Invoke and indefinitely run an iperf server
func iperf_server() {

	output, success := CmdExec(IPERF3_PATH, []string{IPERF3_PATH, "-s", host, "-J", "-i", "60"}, 15)
	if success {
		fmt.Println(output)
	}
}

// Invoke and indefinitely run netperf server
func netperf_server() {

	output, success := CmdExec(NETPERF_SERVER_PATH, []string{NETPERF_SERVER_PATH, "-D"}, 15)
	if success {
		fmt.Println(output)
	}
}

// Invoke and run an iperf client and return the output if successful.
func iperf_client(serverHost, serverPort string, mss int, workItemType int) (rv string) {

	switch {
	case workItemType == IPERF_TCP_TEST:
		output, success := CmdExec(IPERF3_PATH, []string{IPERF3_PATH, "-c", serverHost, "-N", "-i", "30", "-t", "10", "-f", "m", "-w", "512M", "-Z", "-P", PARALLELSTREAMS, "-M", string(mss)}, 15)
		if success {
			rv = output
		}

	case workItemType == IPERF_UDP_TEST:
		output, success := CmdExec(IPERF3_PATH, []string{IPERF3_PATH, "-c", serverHost, "-i", "30", "-t", "10", "-f", "m", "-b", "0", "-u"}, 15)
		if success {
			rv = output
		}
	}
	return
}

// Invoke and run a netperf client and return the output if successful.
func netperf_client(serverHost, serverPort string, workItemType int) (rv string) {

	output, success := CmdExec(NETPERF_PATH, []string{NETPERF_PATH, "-H", serverHost}, 15)
	if success {
		fmt.Println(output)
		rv = output
	} else {
		fmt.Println("Error running netperf client", output)
	}

	return
}

func CmdExec(command string, args []string, timeout int32) (rv string, rc bool) {

	cmd := exec.Cmd{Path: command, Args: args}

	var stdoutput bytes.Buffer
	var stderror bytes.Buffer
	cmd.Stdout = &stdoutput
	cmd.Stderr = &stderror
	if err := cmd.Run(); err != nil {
		outputstr := stdoutput.String()
		errstr := stderror.String()
		fmt.Println("Failed to run", outputstr, "error:", errstr, err)
		return
	}

	rv = stdoutput.String()
	rc = true
	return
}
