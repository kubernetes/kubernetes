// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path"
	"regexp"
	"time"
)

var (
	etcdctlPath    string
	etcdPath       string
	etcdRestoreDir string
	etcdName       string
	etcdPeerUrls   string
)

func main() {
	flag.StringVar(&etcdctlPath, "etcdctl-path", "/usr/bin/etcdctl", "absolute path to etcdctl executable")
	flag.StringVar(&etcdPath, "etcd-path", "/usr/bin/etcd2", "absolute path to etcd2 executable")
	flag.StringVar(&etcdRestoreDir, "etcd-restore-dir", "/var/lib/etcd2-restore", "absolute path to etcd2 restore dir")
	flag.StringVar(&etcdName, "etcd-name", "default", "name of etcd2 node")
	flag.StringVar(&etcdPeerUrls, "etcd-peer-urls", "", "advertise peer urls")

	flag.Parse()

	if etcdPeerUrls == "" {
		panic("must set -etcd-peer-urls")
	}

	if finfo, err := os.Stat(etcdRestoreDir); err != nil {
		panic(err)
	} else {
		if !finfo.IsDir() {
			panic(fmt.Errorf("%s is not a directory", etcdRestoreDir))
		}
	}

	if !path.IsAbs(etcdctlPath) {
		panic(fmt.Sprintf("etcdctl-path %s is not absolute", etcdctlPath))
	}

	if !path.IsAbs(etcdPath) {
		panic(fmt.Sprintf("etcd-path %s is not absolute", etcdPath))
	}

	if err := restoreEtcd(); err != nil {
		panic(err)
	}
}

func restoreEtcd() error {
	etcdCmd := exec.Command(etcdPath, "--force-new-cluster", "--data-dir", etcdRestoreDir)

	etcdCmd.Stdout = os.Stdout
	etcdCmd.Stderr = os.Stderr

	if err := etcdCmd.Start(); err != nil {
		return fmt.Errorf("Could not start etcd2: %s", err)
	}
	defer etcdCmd.Wait()
	defer etcdCmd.Process.Kill()

	return runCommands(10, 2*time.Second)
}

var clusterHealthRegex = regexp.MustCompile(".*cluster is healthy.*")
var lineSplit = regexp.MustCompile("\n+")
var colonSplit = regexp.MustCompile("\\:")

func runCommands(maxRetry int, interval time.Duration) error {
	var retryCnt int
	for retryCnt = 1; retryCnt <= maxRetry; retryCnt++ {
		out, err := exec.Command(etcdctlPath, "cluster-health").CombinedOutput()
		if err == nil && clusterHealthRegex.Match(out) {
			break
		}
		fmt.Printf("Error: %s: %s\n", err, string(out))
		time.Sleep(interval)
	}

	if retryCnt > maxRetry {
		return fmt.Errorf("Timed out waiting for healthy cluster\n")
	}

	var (
		memberID string
		out      []byte
		err      error
	)
	if out, err = exec.Command(etcdctlPath, "member", "list").CombinedOutput(); err != nil {
		return fmt.Errorf("Error calling member list: %s", err)
	}
	members := lineSplit.Split(string(out), 2)
	if len(members) < 1 {
		return fmt.Errorf("Could not find a cluster member from: \"%s\"", members)
	}
	parts := colonSplit.Split(members[0], 2)
	if len(parts) < 2 {
		return fmt.Errorf("Could not parse member id from: \"%s\"", members[0])
	}
	memberID = parts[0]

	out, err = exec.Command(etcdctlPath, "member", "update", memberID, etcdPeerUrls).CombinedOutput()
	fmt.Printf("member update result: %s\n", string(out))
	return err
}
