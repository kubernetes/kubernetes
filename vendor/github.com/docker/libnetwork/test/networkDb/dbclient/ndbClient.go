package dbclient

import (
	"context"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

var servicePort string

const totalWrittenKeys string = "totalKeys"

type resultTuple struct {
	id     string
	result int
}

func httpGetFatalError(ip, port, path string) {
	// for {
	body, err := httpGet(ip, port, path)
	if err != nil || !strings.Contains(string(body), "OK") {
		// if strings.Contains(err.Error(), "EOF") {
		// 	logrus.Warnf("Got EOF path:%s err:%s", path, err)
		// 	continue
		// }
		log.Fatalf("[%s] error %s %s", path, err, body)
	}
	// break
	// }
}

func httpGet(ip, port, path string) ([]byte, error) {
	resp, err := http.Get("http://" + ip + ":" + port + path)
	if err != nil {
		logrus.Errorf("httpGet error:%s", err)
		return nil, err
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	return body, err
}

func joinCluster(ip, port string, members []string, doneCh chan resultTuple) {
	httpGetFatalError(ip, port, "/join?members="+strings.Join(members, ","))

	if doneCh != nil {
		doneCh <- resultTuple{id: ip, result: 0}
	}
}

func joinNetwork(ip, port, network string, doneCh chan resultTuple) {
	httpGetFatalError(ip, port, "/joinnetwork?nid="+network)

	if doneCh != nil {
		doneCh <- resultTuple{id: ip, result: 0}
	}
}

func leaveNetwork(ip, port, network string, doneCh chan resultTuple) {
	httpGetFatalError(ip, port, "/leavenetwork?nid="+network)

	if doneCh != nil {
		doneCh <- resultTuple{id: ip, result: 0}
	}
}

func writeTableKey(ip, port, networkName, tableName, key string) {
	createPath := "/createentry?nid=" + networkName + "&tname=" + tableName + "&value=v&key="
	httpGetFatalError(ip, port, createPath+key)
}

func deleteTableKey(ip, port, networkName, tableName, key string) {
	deletePath := "/deleteentry?nid=" + networkName + "&tname=" + tableName + "&key="
	httpGetFatalError(ip, port, deletePath+key)
}

func clusterPeersNumber(ip, port string, doneCh chan resultTuple) {
	body, err := httpGet(ip, port, "/clusterpeers")

	if err != nil {
		logrus.Errorf("clusterPeers %s there was an error: %s\n", ip, err)
		doneCh <- resultTuple{id: ip, result: -1}
		return
	}
	peersRegexp := regexp.MustCompile(`Total peers: ([0-9]+)`)
	peersNum, _ := strconv.Atoi(peersRegexp.FindStringSubmatch(string(body))[1])

	doneCh <- resultTuple{id: ip, result: peersNum}
}

func networkPeersNumber(ip, port, networkName string, doneCh chan resultTuple) {
	body, err := httpGet(ip, port, "/networkpeers?nid="+networkName)

	if err != nil {
		logrus.Errorf("networkPeersNumber %s there was an error: %s\n", ip, err)
		doneCh <- resultTuple{id: ip, result: -1}
		return
	}
	peersRegexp := regexp.MustCompile(`Total peers: ([0-9]+)`)
	peersNum, _ := strconv.Atoi(peersRegexp.FindStringSubmatch(string(body))[1])

	doneCh <- resultTuple{id: ip, result: peersNum}
}

func dbTableEntriesNumber(ip, port, networkName, tableName string, doneCh chan resultTuple) {
	body, err := httpGet(ip, port, "/gettable?nid="+networkName+"&tname="+tableName)

	if err != nil {
		logrus.Errorf("tableEntriesNumber %s there was an error: %s\n", ip, err)
		doneCh <- resultTuple{id: ip, result: -1}
		return
	}
	elementsRegexp := regexp.MustCompile(`total elements: ([0-9]+)`)
	entriesNum, _ := strconv.Atoi(elementsRegexp.FindStringSubmatch(string(body))[1])
	doneCh <- resultTuple{id: ip, result: entriesNum}
}

func clientWatchTable(ip, port, networkName, tableName string, doneCh chan resultTuple) {
	httpGetFatalError(ip, port, "/watchtable?nid="+networkName+"&tname="+tableName)
	if doneCh != nil {
		doneCh <- resultTuple{id: ip, result: 0}
	}
}

func clientTableEntriesNumber(ip, port, networkName, tableName string, doneCh chan resultTuple) {
	body, err := httpGet(ip, port, "/watchedtableentries?nid="+networkName+"&tname="+tableName)

	if err != nil {
		logrus.Errorf("clientTableEntriesNumber %s there was an error: %s\n", ip, err)
		doneCh <- resultTuple{id: ip, result: -1}
		return
	}
	elementsRegexp := regexp.MustCompile(`total elements: ([0-9]+)`)
	entriesNum, _ := strconv.Atoi(elementsRegexp.FindStringSubmatch(string(body))[1])
	doneCh <- resultTuple{id: ip, result: entriesNum}
}

func writeUniqueKeys(ctx context.Context, ip, port, networkName, tableName, key string, doneCh chan resultTuple) {
	for x := 0; ; x++ {
		select {
		case <-ctx.Done():
			doneCh <- resultTuple{id: ip, result: x}
			return
		default:
			k := key + strconv.Itoa(x)
			// write key
			writeTableKey(ip, port, networkName, tableName, k)
			// give time to send out key writes
			time.Sleep(100 * time.Millisecond)
		}
	}
}

func writeDeleteUniqueKeys(ctx context.Context, ip, port, networkName, tableName, key string, doneCh chan resultTuple) {
	for x := 0; ; x++ {
		select {
		case <-ctx.Done():
			doneCh <- resultTuple{id: ip, result: x}
			return
		default:
			k := key + strconv.Itoa(x)
			// write key
			writeTableKey(ip, port, networkName, tableName, k)
			// give time to send out key writes
			time.Sleep(100 * time.Millisecond)
			// delete key
			deleteTableKey(ip, port, networkName, tableName, k)
		}
	}
}

func writeDeleteLeaveJoin(ctx context.Context, ip, port, networkName, tableName, key string, doneCh chan resultTuple) {
	for x := 0; ; x++ {
		select {
		case <-ctx.Done():
			doneCh <- resultTuple{id: ip, result: x}
			return
		default:
			k := key + strconv.Itoa(x)
			// write key
			writeTableKey(ip, port, networkName, tableName, k)
			time.Sleep(100 * time.Millisecond)
			// delete key
			deleteTableKey(ip, port, networkName, tableName, k)
			// give some time
			time.Sleep(100 * time.Millisecond)
			// leave network
			leaveNetwork(ip, port, networkName, nil)
			// join network
			joinNetwork(ip, port, networkName, nil)
		}
	}
}

func ready(ip, port string, doneCh chan resultTuple) {
	for {
		body, err := httpGet(ip, port, "/ready")
		if err != nil || !strings.Contains(string(body), "OK") {
			time.Sleep(500 * time.Millisecond)
			continue
		}
		// success
		break
	}
	// notify the completion
	doneCh <- resultTuple{id: ip, result: 0}
}

func checkTable(ctx context.Context, ips []string, port, networkName, tableName string, expectedEntries int, fn func(string, string, string, string, chan resultTuple)) {
	startTime := time.Now().UnixNano()
	var successTime int64

	// Loop for 2 minutes to guartee that the result is stable
	for {
		select {
		case <-ctx.Done():
			// Validate test success, if the time is set means that all the tables are empty
			if successTime != 0 {
				logrus.Infof("Check table passed, the cluster converged in %d msec", time.Duration(successTime-startTime)/time.Millisecond)
				return
			}
			log.Fatal("Test failed, there is still entries in the tables of the nodes")
		default:
			logrus.Infof("Checking table %s expected %d", tableName, expectedEntries)
			doneCh := make(chan resultTuple, len(ips))
			for _, ip := range ips {
				go fn(ip, servicePort, networkName, tableName, doneCh)
			}

			nodesWithCorrectEntriesNum := 0
			for i := len(ips); i > 0; i-- {
				tableEntries := <-doneCh
				logrus.Infof("Node %s has %d entries", tableEntries.id, tableEntries.result)
				if tableEntries.result == expectedEntries {
					nodesWithCorrectEntriesNum++
				}
			}
			close(doneCh)
			if nodesWithCorrectEntriesNum == len(ips) {
				if successTime == 0 {
					successTime = time.Now().UnixNano()
					logrus.Infof("Success after %d msec", time.Duration(successTime-startTime)/time.Millisecond)
				}
			} else {
				successTime = 0
			}
			time.Sleep(10 * time.Second)
		}
	}
}

func waitWriters(parallelWriters int, mustWrite bool, doneCh chan resultTuple) map[string]int {
	var totalKeys int
	resultTable := make(map[string]int)
	for i := 0; i < parallelWriters; i++ {
		logrus.Infof("Waiting for %d workers", parallelWriters-i)
		workerReturn := <-doneCh
		totalKeys += workerReturn.result
		if mustWrite && workerReturn.result == 0 {
			log.Fatalf("The worker %s did not write any key %d == 0", workerReturn.id, workerReturn.result)
		}
		if !mustWrite && workerReturn.result != 0 {
			log.Fatalf("The worker %s was supposed to return 0 instead %d != 0", workerReturn.id, workerReturn.result)
		}
		if mustWrite {
			resultTable[workerReturn.id] = workerReturn.result
			logrus.Infof("The worker %s wrote %d keys", workerReturn.id, workerReturn.result)
		}
	}
	resultTable[totalWrittenKeys] = totalKeys
	return resultTable
}

// ready
func doReady(ips []string) {
	doneCh := make(chan resultTuple, len(ips))
	// check all the nodes
	for _, ip := range ips {
		go ready(ip, servicePort, doneCh)
	}
	// wait for the readiness of all nodes
	for i := len(ips); i > 0; i-- {
		<-doneCh
	}
	close(doneCh)
}

// join
func doJoin(ips []string) {
	doneCh := make(chan resultTuple, len(ips))
	// check all the nodes
	for i, ip := range ips {
		members := append([]string(nil), ips[:i]...)
		members = append(members, ips[i+1:]...)
		go joinCluster(ip, servicePort, members, doneCh)
	}
	// wait for the readiness of all nodes
	for i := len(ips); i > 0; i-- {
		<-doneCh
	}
	close(doneCh)
}

// cluster-peers expectedNumberPeers
func doClusterPeers(ips []string, args []string) {
	doneCh := make(chan resultTuple, len(ips))
	expectedPeers, _ := strconv.Atoi(args[0])
	// check all the nodes
	for _, ip := range ips {
		go clusterPeersNumber(ip, servicePort, doneCh)
	}
	// wait for the readiness of all nodes
	for i := len(ips); i > 0; i-- {
		node := <-doneCh
		if node.result != expectedPeers {
			log.Fatalf("Expected peers from %s missmatch %d != %d", node.id, expectedPeers, node.result)
		}
	}
	close(doneCh)
}

// join-network networkName
func doJoinNetwork(ips []string, args []string) {
	doneCh := make(chan resultTuple, len(ips))
	// check all the nodes
	for _, ip := range ips {
		go joinNetwork(ip, servicePort, args[0], doneCh)
	}
	// wait for the readiness of all nodes
	for i := len(ips); i > 0; i-- {
		<-doneCh
	}
	close(doneCh)
}

// leave-network networkName
func doLeaveNetwork(ips []string, args []string) {
	doneCh := make(chan resultTuple, len(ips))
	// check all the nodes
	for _, ip := range ips {
		go leaveNetwork(ip, servicePort, args[0], doneCh)
	}
	// wait for the readiness of all nodes
	for i := len(ips); i > 0; i-- {
		<-doneCh
	}
	close(doneCh)
}

// cluster-peers networkName expectedNumberPeers maxRetry
func doNetworkPeers(ips []string, args []string) {
	doneCh := make(chan resultTuple, len(ips))
	networkName := args[0]
	expectedPeers, _ := strconv.Atoi(args[1])
	maxRetry, _ := strconv.Atoi(args[2])
	for retry := 0; retry < maxRetry; retry++ {
		// check all the nodes
		for _, ip := range ips {
			go networkPeersNumber(ip, servicePort, networkName, doneCh)
		}
		// wait for the readiness of all nodes
		for i := len(ips); i > 0; i-- {
			node := <-doneCh
			if node.result != expectedPeers {
				if retry == maxRetry-1 {
					log.Fatalf("Expected peers from %s missmatch %d != %d", node.id, expectedPeers, node.result)
				} else {
					logrus.Warnf("Expected peers from %s missmatch %d != %d", node.id, expectedPeers, node.result)
				}
				time.Sleep(1 * time.Second)
			}
		}
	}
	close(doneCh)
}

// write-delete-unique-keys networkName tableName numParallelWriters writeTimeSec
func doWriteDeleteUniqueKeys(ips []string, args []string) {
	networkName := args[0]
	tableName := args[1]
	parallelWriters, _ := strconv.Atoi(args[2])
	writeTimeSec, _ := strconv.Atoi(args[3])

	doneCh := make(chan resultTuple, parallelWriters)
	// Enable watch of tables from clients
	for i := 0; i < parallelWriters; i++ {
		go clientWatchTable(ips[i], servicePort, networkName, tableName, doneCh)
	}
	waitWriters(parallelWriters, false, doneCh)

	// Start parallel writers that will create and delete unique keys
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(writeTimeSec)*time.Second)
	for i := 0; i < parallelWriters; i++ {
		key := "key-" + strconv.Itoa(i) + "-"
		logrus.Infof("Spawn worker: %d on IP:%s", i, ips[i])
		go writeDeleteUniqueKeys(ctx, ips[i], servicePort, networkName, tableName, key, doneCh)
	}

	// Sync with all the writers
	keyMap := waitWriters(parallelWriters, true, doneCh)
	cancel()
	logrus.Infof("Written a total of %d keys on the cluster", keyMap[totalWrittenKeys])

	// check table entries for 2 minutes
	ctx, cancel = context.WithTimeout(context.Background(), 2*time.Minute)
	checkTable(ctx, ips, servicePort, networkName, tableName, 0, dbTableEntriesNumber)
	cancel()
	ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
	checkTable(ctx, ips, servicePort, networkName, tableName, 0, clientTableEntriesNumber)
	cancel()
}

// write-unique-keys networkName tableName numParallelWriters writeTimeSec
func doWriteUniqueKeys(ips []string, args []string) {
	networkName := args[0]
	tableName := args[1]
	parallelWriters, _ := strconv.Atoi(args[2])
	writeTimeSec, _ := strconv.Atoi(args[3])

	doneCh := make(chan resultTuple, parallelWriters)
	// Enable watch of tables from clients
	for i := 0; i < parallelWriters; i++ {
		go clientWatchTable(ips[i], servicePort, networkName, tableName, doneCh)
	}
	waitWriters(parallelWriters, false, doneCh)

	// Start parallel writers that will create and delete unique keys
	defer close(doneCh)
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(writeTimeSec)*time.Second)
	for i := 0; i < parallelWriters; i++ {
		key := "key-" + strconv.Itoa(i) + "-"
		logrus.Infof("Spawn worker: %d on IP:%s", i, ips[i])
		go writeUniqueKeys(ctx, ips[i], servicePort, networkName, tableName, key, doneCh)
	}

	// Sync with all the writers
	keyMap := waitWriters(parallelWriters, true, doneCh)
	cancel()
	logrus.Infof("Written a total of %d keys on the cluster", keyMap[totalWrittenKeys])

	// check table entries for 2 minutes
	ctx, cancel = context.WithTimeout(context.Background(), 2*time.Minute)
	checkTable(ctx, ips, servicePort, networkName, tableName, keyMap[totalWrittenKeys], dbTableEntriesNumber)
	cancel()
}

// write-delete-leave-join networkName tableName numParallelWriters writeTimeSec
func doWriteDeleteLeaveJoin(ips []string, args []string) {
	networkName := args[0]
	tableName := args[1]
	parallelWriters, _ := strconv.Atoi(args[2])
	writeTimeSec, _ := strconv.Atoi(args[3])

	// Start parallel writers that will create and delete unique keys
	doneCh := make(chan resultTuple, parallelWriters)
	defer close(doneCh)
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(writeTimeSec)*time.Second)
	for i := 0; i < parallelWriters; i++ {
		key := "key-" + strconv.Itoa(i) + "-"
		logrus.Infof("Spawn worker: %d on IP:%s", i, ips[i])
		go writeDeleteLeaveJoin(ctx, ips[i], servicePort, networkName, tableName, key, doneCh)
	}

	// Sync with all the writers
	keyMap := waitWriters(parallelWriters, true, doneCh)
	cancel()
	logrus.Infof("Written a total of %d keys on the cluster", keyMap["totalKeys"])

	// check table entries for 2 minutes
	ctx, cancel = context.WithTimeout(context.Background(), 2*time.Minute)
	checkTable(ctx, ips, servicePort, networkName, tableName, 0, dbTableEntriesNumber)
	cancel()
}

// write-delete-wait-leave-join networkName tableName numParallelWriters writeTimeSec
func doWriteDeleteWaitLeaveJoin(ips []string, args []string) {
	networkName := args[0]
	tableName := args[1]
	parallelWriters, _ := strconv.Atoi(args[2])
	writeTimeSec, _ := strconv.Atoi(args[3])

	// Start parallel writers that will create and delete unique keys
	doneCh := make(chan resultTuple, parallelWriters)
	defer close(doneCh)
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(writeTimeSec)*time.Second)
	for i := 0; i < parallelWriters; i++ {
		key := "key-" + strconv.Itoa(i) + "-"
		logrus.Infof("Spawn worker: %d on IP:%s", i, ips[i])
		go writeDeleteUniqueKeys(ctx, ips[i], servicePort, networkName, tableName, key, doneCh)
	}

	// Sync with all the writers
	keyMap := waitWriters(parallelWriters, true, doneCh)
	cancel()
	logrus.Infof("Written a total of %d keys on the cluster", keyMap[totalWrittenKeys])

	// The writers will leave the network
	for i := 0; i < parallelWriters; i++ {
		logrus.Infof("worker leaveNetwork: %d on IP:%s", i, ips[i])
		go leaveNetwork(ips[i], servicePort, networkName, doneCh)
	}
	waitWriters(parallelWriters, false, doneCh)

	// Give some time
	time.Sleep(100 * time.Millisecond)

	// The writers will join the network
	for i := 0; i < parallelWriters; i++ {
		logrus.Infof("worker joinNetwork: %d on IP:%s", i, ips[i])
		go joinNetwork(ips[i], servicePort, networkName, doneCh)
	}
	waitWriters(parallelWriters, false, doneCh)

	// check table entries for 2 minutes
	ctx, cancel = context.WithTimeout(context.Background(), 2*time.Minute)
	checkTable(ctx, ips, servicePort, networkName, tableName, 0, dbTableEntriesNumber)
	cancel()
}

// write-wait-leave networkName tableName numParallelWriters writeTimeSec
func doWriteWaitLeave(ips []string, args []string) {
	networkName := args[0]
	tableName := args[1]
	parallelWriters, _ := strconv.Atoi(args[2])
	writeTimeSec, _ := strconv.Atoi(args[3])

	// Start parallel writers that will create and delete unique keys
	doneCh := make(chan resultTuple, parallelWriters)
	defer close(doneCh)
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(writeTimeSec)*time.Second)
	for i := 0; i < parallelWriters; i++ {
		key := "key-" + strconv.Itoa(i) + "-"
		logrus.Infof("Spawn worker: %d on IP:%s", i, ips[i])
		go writeUniqueKeys(ctx, ips[i], servicePort, networkName, tableName, key, doneCh)
	}

	// Sync with all the writers
	keyMap := waitWriters(parallelWriters, true, doneCh)
	cancel()
	logrus.Infof("Written a total of %d keys on the cluster", keyMap[totalWrittenKeys])

	// The writers will leave the network
	for i := 0; i < parallelWriters; i++ {
		logrus.Infof("worker leaveNetwork: %d on IP:%s", i, ips[i])
		go leaveNetwork(ips[i], servicePort, networkName, doneCh)
	}
	waitWriters(parallelWriters, false, doneCh)

	// check table entries for 2 minutes
	ctx, cancel = context.WithTimeout(context.Background(), 2*time.Minute)
	checkTable(ctx, ips, servicePort, networkName, tableName, 0, dbTableEntriesNumber)
	cancel()
}

// write-wait-leave-join networkName tableName numParallelWriters writeTimeSec numParallelLeaver
func doWriteWaitLeaveJoin(ips []string, args []string) {
	networkName := args[0]
	tableName := args[1]
	parallelWriters, _ := strconv.Atoi(args[2])
	writeTimeSec, _ := strconv.Atoi(args[3])
	parallerlLeaver, _ := strconv.Atoi(args[4])

	// Start parallel writers that will create and delete unique keys
	doneCh := make(chan resultTuple, parallelWriters)
	defer close(doneCh)
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(writeTimeSec)*time.Second)
	for i := 0; i < parallelWriters; i++ {
		key := "key-" + strconv.Itoa(i) + "-"
		logrus.Infof("Spawn worker: %d on IP:%s", i, ips[i])
		go writeUniqueKeys(ctx, ips[i], servicePort, networkName, tableName, key, doneCh)
	}

	// Sync with all the writers
	keyMap := waitWriters(parallelWriters, true, doneCh)
	cancel()
	logrus.Infof("Written a total of %d keys on the cluster", keyMap[totalWrittenKeys])

	keysExpected := keyMap[totalWrittenKeys]
	// The Leavers will leave the network
	for i := 0; i < parallerlLeaver; i++ {
		logrus.Infof("worker leaveNetwork: %d on IP:%s", i, ips[i])
		go leaveNetwork(ips[i], servicePort, networkName, doneCh)
		// Once a node leave all the keys written previously will be deleted, so the expected keys will consider that as removed
		keysExpected -= keyMap[ips[i]]
	}
	waitWriters(parallerlLeaver, false, doneCh)

	// Give some time
	time.Sleep(100 * time.Millisecond)

	// The writers will join the network
	for i := 0; i < parallerlLeaver; i++ {
		logrus.Infof("worker joinNetwork: %d on IP:%s", i, ips[i])
		go joinNetwork(ips[i], servicePort, networkName, doneCh)
	}
	waitWriters(parallerlLeaver, false, doneCh)

	// check table entries for 2 minutes
	ctx, cancel = context.WithTimeout(context.Background(), 2*time.Minute)
	checkTable(ctx, ips, servicePort, networkName, tableName, keysExpected, dbTableEntriesNumber)
	cancel()
}

var cmdArgChec = map[string]int{
	"debug":                    0,
	"fail":                     0,
	"ready":                    2,
	"join":                     2,
	"leave":                    2,
	"join-network":             3,
	"leave-network":            3,
	"cluster-peers":            3,
	"write-delete-unique-keys": 4,
}

// Client is a client
func Client(args []string) {
	logrus.Infof("[CLIENT] Starting with arguments %v", args)
	command := args[0]

	if len(args) < cmdArgChec[command] {
		log.Fatalf("Command %s requires %d arguments, aborting...", command, cmdArgChec[command])
	}

	switch command {
	case "debug":
		time.Sleep(1 * time.Hour)
		os.Exit(0)
	case "fail":
		log.Fatalf("Test error condition with message: error error error")
	}

	serviceName := args[1]
	ips, _ := net.LookupHost("tasks." + serviceName)
	logrus.Infof("got the ips %v", ips)
	if len(ips) == 0 {
		log.Fatalf("Cannot resolve any IP for the service tasks.%s", serviceName)
	}
	servicePort = args[2]
	commandArgs := args[3:]
	logrus.Infof("Executing %s with args:%v", command, commandArgs)
	switch command {
	case "ready":
		doReady(ips)
	case "join":
		doJoin(ips)
	case "leave":

	case "cluster-peers":
		// cluster-peers
		doClusterPeers(ips, commandArgs)

	case "join-network":
		// join-network networkName
		doJoinNetwork(ips, commandArgs)
	case "leave-network":
		// leave-network networkName
		doLeaveNetwork(ips, commandArgs)
	case "network-peers":
		// network-peers networkName maxRetry
		doNetworkPeers(ips, commandArgs)

	case "write-unique-keys":
		// write-delete-unique-keys networkName tableName numParallelWriters writeTimeSec
		doWriteUniqueKeys(ips, commandArgs)
	case "write-delete-unique-keys":
		// write-delete-unique-keys networkName tableName numParallelWriters writeTimeSec
		doWriteDeleteUniqueKeys(ips, commandArgs)
	case "write-delete-leave-join":
		// write-delete-leave-join networkName tableName numParallelWriters writeTimeSec
		doWriteDeleteLeaveJoin(ips, commandArgs)
	case "write-delete-wait-leave-join":
		// write-delete-wait-leave-join networkName tableName numParallelWriters writeTimeSec
		doWriteDeleteWaitLeaveJoin(ips, commandArgs)
	case "write-wait-leave":
		// write-wait-leave networkName tableName numParallelWriters writeTimeSec
		doWriteWaitLeave(ips, commandArgs)
	case "write-wait-leave-join":
		// write-wait-leave networkName tableName numParallelWriters writeTimeSec
		doWriteWaitLeaveJoin(ips, commandArgs)
	default:
		log.Fatalf("Command %s not recognized", command)
	}
}
