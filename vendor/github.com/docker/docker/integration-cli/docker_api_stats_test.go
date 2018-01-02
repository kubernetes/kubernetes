package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/versions"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/request"
	"github.com/go-check/check"
)

var expectedNetworkInterfaceStats = strings.Split("rx_bytes rx_dropped rx_errors rx_packets tx_bytes tx_dropped tx_errors tx_packets", " ")

func (s *DockerSuite) TestAPIStatsNoStreamGetCpu(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "while true;usleep 100; do echo 'Hello'; done")

	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), checker.IsNil)
	resp, body, err := request.Get(fmt.Sprintf("/containers/%s/stats?stream=false", id))
	c.Assert(err, checker.IsNil)
	c.Assert(resp.StatusCode, checker.Equals, http.StatusOK)
	c.Assert(resp.Header.Get("Content-Type"), checker.Equals, "application/json")

	var v *types.Stats
	err = json.NewDecoder(body).Decode(&v)
	c.Assert(err, checker.IsNil)
	body.Close()

	var cpuPercent = 0.0

	if testEnv.DaemonPlatform() != "windows" {
		cpuDelta := float64(v.CPUStats.CPUUsage.TotalUsage - v.PreCPUStats.CPUUsage.TotalUsage)
		systemDelta := float64(v.CPUStats.SystemUsage - v.PreCPUStats.SystemUsage)
		cpuPercent = (cpuDelta / systemDelta) * float64(len(v.CPUStats.CPUUsage.PercpuUsage)) * 100.0
	} else {
		// Max number of 100ns intervals between the previous time read and now
		possIntervals := uint64(v.Read.Sub(v.PreRead).Nanoseconds()) // Start with number of ns intervals
		possIntervals /= 100                                         // Convert to number of 100ns intervals
		possIntervals *= uint64(v.NumProcs)                          // Multiple by the number of processors

		// Intervals used
		intervalsUsed := v.CPUStats.CPUUsage.TotalUsage - v.PreCPUStats.CPUUsage.TotalUsage

		// Percentage avoiding divide-by-zero
		if possIntervals > 0 {
			cpuPercent = float64(intervalsUsed) / float64(possIntervals) * 100.0
		}
	}

	c.Assert(cpuPercent, check.Not(checker.Equals), 0.0, check.Commentf("docker stats with no-stream get cpu usage failed: was %v", cpuPercent))
}

func (s *DockerSuite) TestAPIStatsStoppedContainerInGoroutines(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "echo 1")
	id := strings.TrimSpace(out)

	getGoRoutines := func() int {
		_, body, err := request.Get(fmt.Sprintf("/info"))
		c.Assert(err, checker.IsNil)
		info := types.Info{}
		err = json.NewDecoder(body).Decode(&info)
		c.Assert(err, checker.IsNil)
		body.Close()
		return info.NGoroutines
	}

	// When the HTTP connection is closed, the number of goroutines should not increase.
	routines := getGoRoutines()
	_, body, err := request.Get(fmt.Sprintf("/containers/%s/stats", id))
	c.Assert(err, checker.IsNil)
	body.Close()

	t := time.After(30 * time.Second)
	for {
		select {
		case <-t:
			c.Assert(getGoRoutines(), checker.LessOrEqualThan, routines)
			return
		default:
			if n := getGoRoutines(); n <= routines {
				return
			}
			time.Sleep(200 * time.Millisecond)
		}
	}
}

func (s *DockerSuite) TestAPIStatsNetworkStats(c *check.C) {
	testRequires(c, SameHostDaemon)

	out := runSleepingContainer(c)
	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), checker.IsNil)

	// Retrieve the container address
	net := "bridge"
	if testEnv.DaemonPlatform() == "windows" {
		net = "nat"
	}
	contIP := findContainerIP(c, id, net)
	numPings := 1

	var preRxPackets uint64
	var preTxPackets uint64
	var postRxPackets uint64
	var postTxPackets uint64

	// Get the container networking stats before and after pinging the container
	nwStatsPre := getNetworkStats(c, id)
	for _, v := range nwStatsPre {
		preRxPackets += v.RxPackets
		preTxPackets += v.TxPackets
	}

	countParam := "-c"
	if runtime.GOOS == "windows" {
		countParam = "-n" // Ping count parameter is -n on Windows
	}
	pingout, err := exec.Command("ping", contIP, countParam, strconv.Itoa(numPings)).CombinedOutput()
	if err != nil && runtime.GOOS == "linux" {
		// If it fails then try a work-around, but just for linux.
		// If this fails too then go back to the old error for reporting.
		//
		// The ping will sometimes fail due to an apparmor issue where it
		// denies access to the libc.so.6 shared library - running it
		// via /lib64/ld-linux-x86-64.so.2 seems to work around it.
		pingout2, err2 := exec.Command("/lib64/ld-linux-x86-64.so.2", "/bin/ping", contIP, "-c", strconv.Itoa(numPings)).CombinedOutput()
		if err2 == nil {
			pingout = pingout2
			err = err2
		}
	}
	c.Assert(err, checker.IsNil)
	pingouts := string(pingout[:])
	nwStatsPost := getNetworkStats(c, id)
	for _, v := range nwStatsPost {
		postRxPackets += v.RxPackets
		postTxPackets += v.TxPackets
	}

	// Verify the stats contain at least the expected number of packets
	// On Linux, account for ARP.
	expRxPkts := preRxPackets + uint64(numPings)
	expTxPkts := preTxPackets + uint64(numPings)
	if testEnv.DaemonPlatform() != "windows" {
		expRxPkts++
		expTxPkts++
	}
	c.Assert(postTxPackets, checker.GreaterOrEqualThan, expTxPkts,
		check.Commentf("Reported less TxPackets than expected. Expected >= %d. Found %d. %s", expTxPkts, postTxPackets, pingouts))
	c.Assert(postRxPackets, checker.GreaterOrEqualThan, expRxPkts,
		check.Commentf("Reported less RxPackets than expected. Expected >= %d. Found %d. %s", expRxPkts, postRxPackets, pingouts))
}

func (s *DockerSuite) TestAPIStatsNetworkStatsVersioning(c *check.C) {
	// Windows doesn't support API versions less than 1.25, so no point testing 1.17 .. 1.21
	testRequires(c, SameHostDaemon, DaemonIsLinux)

	out := runSleepingContainer(c)
	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), checker.IsNil)
	wg := sync.WaitGroup{}

	for i := 17; i <= 21; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			apiVersion := fmt.Sprintf("v1.%d", i)
			statsJSONBlob := getVersionedStats(c, id, apiVersion)
			if versions.LessThan(apiVersion, "v1.21") {
				c.Assert(jsonBlobHasLTv121NetworkStats(statsJSONBlob), checker.Equals, true,
					check.Commentf("Stats JSON blob from API %s %#v does not look like a <v1.21 API stats structure", apiVersion, statsJSONBlob))
			} else {
				c.Assert(jsonBlobHasGTE121NetworkStats(statsJSONBlob), checker.Equals, true,
					check.Commentf("Stats JSON blob from API %s %#v does not look like a >=v1.21 API stats structure", apiVersion, statsJSONBlob))
			}
		}(i)
	}
	wg.Wait()
}

func getNetworkStats(c *check.C, id string) map[string]types.NetworkStats {
	var st *types.StatsJSON

	_, body, err := request.Get(fmt.Sprintf("/containers/%s/stats?stream=false", id))
	c.Assert(err, checker.IsNil)

	err = json.NewDecoder(body).Decode(&st)
	c.Assert(err, checker.IsNil)
	body.Close()

	return st.Networks
}

// getVersionedStats returns stats result for the
// container with id using an API call with version apiVersion. Since the
// stats result type differs between API versions, we simply return
// map[string]interface{}.
func getVersionedStats(c *check.C, id string, apiVersion string) map[string]interface{} {
	stats := make(map[string]interface{})

	_, body, err := request.Get(fmt.Sprintf("/%s/containers/%s/stats?stream=false", apiVersion, id))
	c.Assert(err, checker.IsNil)
	defer body.Close()

	err = json.NewDecoder(body).Decode(&stats)
	c.Assert(err, checker.IsNil, check.Commentf("failed to decode stat: %s", err))

	return stats
}

func jsonBlobHasLTv121NetworkStats(blob map[string]interface{}) bool {
	networkStatsIntfc, ok := blob["network"]
	if !ok {
		return false
	}
	networkStats, ok := networkStatsIntfc.(map[string]interface{})
	if !ok {
		return false
	}
	for _, expectedKey := range expectedNetworkInterfaceStats {
		if _, ok := networkStats[expectedKey]; !ok {
			return false
		}
	}
	return true
}

func jsonBlobHasGTE121NetworkStats(blob map[string]interface{}) bool {
	networksStatsIntfc, ok := blob["networks"]
	if !ok {
		return false
	}
	networksStats, ok := networksStatsIntfc.(map[string]interface{})
	if !ok {
		return false
	}
	for _, networkInterfaceStatsIntfc := range networksStats {
		networkInterfaceStats, ok := networkInterfaceStatsIntfc.(map[string]interface{})
		if !ok {
			return false
		}
		for _, expectedKey := range expectedNetworkInterfaceStats {
			if _, ok := networkInterfaceStats[expectedKey]; !ok {
				return false
			}
		}
	}
	return true
}

func (s *DockerSuite) TestAPIStatsContainerNotFound(c *check.C) {
	testRequires(c, DaemonIsLinux)

	status, _, err := request.SockRequest("GET", "/containers/nonexistent/stats", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNotFound)

	status, _, err = request.SockRequest("GET", "/containers/nonexistent/stats?stream=0", nil, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNotFound)
}

func (s *DockerSuite) TestAPIStatsNoStreamConnectedContainers(c *check.C) {
	testRequires(c, DaemonIsLinux)

	out1 := runSleepingContainer(c)
	id1 := strings.TrimSpace(out1)
	c.Assert(waitRun(id1), checker.IsNil)

	out2 := runSleepingContainer(c, "--net", "container:"+id1)
	id2 := strings.TrimSpace(out2)
	c.Assert(waitRun(id2), checker.IsNil)

	ch := make(chan error)
	go func() {
		resp, body, err := request.Get(fmt.Sprintf("/containers/%s/stats?stream=false", id2))
		defer body.Close()
		if err != nil {
			ch <- err
		}
		if resp.StatusCode != http.StatusOK {
			ch <- fmt.Errorf("Invalid StatusCode %v", resp.StatusCode)
		}
		if resp.Header.Get("Content-Type") != "application/json" {
			ch <- fmt.Errorf("Invalid 'Content-Type' %v", resp.Header.Get("Content-Type"))
		}
		var v *types.Stats
		if err := json.NewDecoder(body).Decode(&v); err != nil {
			ch <- err
		}
		ch <- nil
	}()

	select {
	case err := <-ch:
		c.Assert(err, checker.IsNil, check.Commentf("Error in stats Engine API: %v", err))
	case <-time.After(15 * time.Second):
		c.Fatalf("Stats did not return after timeout")
	}
}
