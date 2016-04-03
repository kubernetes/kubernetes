package main

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestCliStatsNoStreamGetCpu(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "while true;do echo 'Hello'; usleep 100000; done")

	id := strings.TrimSpace(out)
	err := waitRun(id)
	c.Assert(err, check.IsNil)

	resp, body, err := sockRequestRaw("GET", fmt.Sprintf("/containers/%s/stats?stream=false", id), nil, "")
	c.Assert(err, check.IsNil)
	c.Assert(resp.ContentLength > 0, check.Equals, true, check.Commentf("should not use chunked encoding"))
	c.Assert(resp.Header.Get("Content-Type"), check.Equals, "application/json")

	var v *types.Stats
	err = json.NewDecoder(body).Decode(&v)
	c.Assert(err, check.IsNil)

	var cpuPercent = 0.0
	cpuDelta := float64(v.CpuStats.CpuUsage.TotalUsage - v.PreCpuStats.CpuUsage.TotalUsage)
	systemDelta := float64(v.CpuStats.SystemUsage - v.PreCpuStats.SystemUsage)
	cpuPercent = (cpuDelta / systemDelta) * float64(len(v.CpuStats.CpuUsage.PercpuUsage)) * 100.0
	if cpuPercent == 0 {
		c.Fatalf("docker stats with no-stream get cpu usage failed: was %v", cpuPercent)
	}
}

func (s *DockerSuite) TestStoppedContainerStatsGoroutines(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "echo 1")
	id := strings.TrimSpace(out)

	getGoRoutines := func() int {
		_, body, err := sockRequestRaw("GET", fmt.Sprintf("/info"), nil, "")
		c.Assert(err, check.IsNil)
		info := types.Info{}
		err = json.NewDecoder(body).Decode(&info)
		c.Assert(err, check.IsNil)
		body.Close()
		return info.NGoroutines
	}

	// When the HTTP connection is closed, the number of goroutines should not increase.
	routines := getGoRoutines()
	_, body, err := sockRequestRaw("GET", fmt.Sprintf("/containers/%s/stats", id), nil, "")
	c.Assert(err, check.IsNil)
	body.Close()

	t := time.After(30 * time.Second)
	for {
		select {
		case <-t:
			c.Assert(getGoRoutines() <= routines, check.Equals, true)
			return
		default:
			if n := getGoRoutines(); n <= routines {
				return
			}
			time.Sleep(200 * time.Millisecond)
		}
	}
}

func (s *DockerSuite) TestApiNetworkStats(c *check.C) {
	testRequires(c, SameHostDaemon)
	// Run container for 30 secs
	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")
	id := strings.TrimSpace(out)
	err := waitRun(id)
	c.Assert(err, check.IsNil)

	// Retrieve the container address
	contIP := findContainerIP(c, id)
	numPings := 10

	// Get the container networking stats before and after pinging the container
	nwStatsPre := getNetworkStats(c, id)
	countParam := "-c"
	if runtime.GOOS == "windows" {
		countParam = "-n" // Ping count parameter is -n on Windows
	}
	pingout, err := exec.Command("ping", contIP, countParam, strconv.Itoa(numPings)).Output()
	pingouts := string(pingout[:])
	c.Assert(err, check.IsNil)
	nwStatsPost := getNetworkStats(c, id)

	// Verify the stats contain at least the expected number of packets (account for ARP)
	expRxPkts := 1 + nwStatsPre.RxPackets + uint64(numPings)
	expTxPkts := 1 + nwStatsPre.TxPackets + uint64(numPings)
	c.Assert(nwStatsPost.TxPackets >= expTxPkts, check.Equals, true,
		check.Commentf("Reported less TxPackets than expected. Expected >= %d. Found %d. %s", expTxPkts, nwStatsPost.TxPackets, pingouts))
	c.Assert(nwStatsPost.RxPackets >= expRxPkts, check.Equals, true,
		check.Commentf("Reported less Txbytes than expected. Expected >= %d. Found %d. %s", expRxPkts, nwStatsPost.RxPackets, pingouts))
}

func getNetworkStats(c *check.C, id string) types.Network {
	var st *types.Stats

	_, body, err := sockRequestRaw("GET", fmt.Sprintf("/containers/%s/stats?stream=false", id), nil, "")
	c.Assert(err, check.IsNil)

	err = json.NewDecoder(body).Decode(&st)
	c.Assert(err, check.IsNil)

	return st.Network
}
