// Copyright 2016 The rkt Authors
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

// +build host coreos src kvm

package main

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	sd_dbus "github.com/coreos/go-systemd/dbus"
	sd_util "github.com/coreos/go-systemd/util"
	"github.com/coreos/rkt/tests/testutils"
	"github.com/vishvananda/netlink"
)

func TestSocketProxyd(t *testing.T) {
	// Skip the test for now. See
	// https://github.com/coreos/rkt/issues/2432#issuecomment-238858840 for
	// details.
	t.Skip("this test is racy, let's skip it until we fix it")

	if !sd_util.IsRunningSystemd() {
		t.Skip("Systemd is not running on the host.")
	}

	socketProxydPath := "/lib/systemd/systemd-socket-proxyd"
	if _, err := os.Stat(socketProxydPath); os.IsNotExist(err) {
		t.Skip("systemd-socket-proxyd is not installed.")
	}

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	iface, _, err := testutils.GetNonLoIfaceWithAddrs(netlink.FAMILY_V4)
	if err != nil {
		t.Fatalf("Error while getting non-lo host interface: %v\n", err)
	}
	if iface.Name == "" {
		t.Skipf("Cannot run test without non-lo host interface")
	}

	nt := networkTemplateT{
		Name:   "ptp0",
		Type:   "ptp",
		IpMasq: true,
		Master: iface.Name,
		Ipam: &ipamTemplateT{
			Type:   "host-local",
			Subnet: "192.168.0.0/24",
			Routes: []map[string]string{
				{"dst": "0.0.0.0/0"},
			},
		},
	}

	netDir := prepareTestNet(t, ctx, nt)
	defer os.RemoveAll(netDir)

	port, err := randomFreePort(t)
	if err != nil {
		t.Fatal(err)
	}

	echoImage := patchTestACI("rkt-inspect-echo.aci",
		"--exec=/echo-socket-activated",
		"--ports=test-port,protocol=tcp,port=80,socketActivated=true")
	defer os.Remove(echoImage)

	conn, err := sd_dbus.New()
	if err != nil {
		t.Fatal(err)
	}

	rktTestingEchoService := `
	[Unit]
	Description=Socket-activated echo server

	[Service]
	ExecStart=%s
	KillMode=process
	`

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	rnd := r.Int()

	// Write unit files directly to runtime system units directory
	// (/run/systemd/system) to avoid calling LinkUnitFiles - it is buggy in
	// systemd v219 as it does not work with absolute paths.
	unitsDir := "/run/systemd/system"
	containerIP := "192.168.0.101"

	cmd := fmt.Sprintf("%s --insecure-options=image --debug run --net=%s:IP=%s --port=test-port:%d --mds-register=false %s",
		ctx.Cmd(), nt.Name, containerIP, port, echoImage)

	serviceContent := fmt.Sprintf(rktTestingEchoService, cmd)
	serviceTargetBase := fmt.Sprintf("rkt-testing-socket-activation-%d.service", rnd)
	serviceTarget := filepath.Join(unitsDir, serviceTargetBase)

	if err := ioutil.WriteFile(serviceTarget, []byte(serviceContent), 0666); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(serviceTarget)

	rktTestingEchoSocket := `
	[Unit]
	Description=Socket-activated netcat server socket

	[Socket]
	ListenStream=%d

	[Install]
	WantedBy=sockets.target
	`

	socketContent := fmt.Sprintf(rktTestingEchoSocket, port)
	socketTargetBase := fmt.Sprintf("proxy-to-rkt-testing-socket-activation-%d.socket", rnd)
	socketTarget := filepath.Join(unitsDir, socketTargetBase)

	if err := ioutil.WriteFile(socketTarget, []byte(socketContent), 0666); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(socketTarget)

	proxyToRktTestingEchoService := `
	[Unit]
	Requires=%s
	After=%s

	[Service]
	ExecStart=%s %s:%d
	`

	proxyContent := fmt.Sprintf(proxyToRktTestingEchoService, serviceTargetBase, serviceTargetBase,
		socketProxydPath, containerIP, port)
	proxyContentBase := fmt.Sprintf("proxy-to-rkt-testing-socket-activation-%d.service", rnd)
	proxyTarget := filepath.Join(unitsDir, proxyContentBase)

	if err := ioutil.WriteFile(proxyTarget, []byte(proxyContent), 0666); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(proxyTarget)

	reschan := make(chan string)
	doJob := func() {
		job := <-reschan
		if job != "done" {
			t.Fatal("Job is not done:", job)
		}
	}

	if _, err := conn.StartUnit(socketTargetBase, "replace", reschan); err != nil {
		t.Fatal(err)
	}
	doJob()

	defer func() {
		if _, err := conn.StopUnit(socketTargetBase, "replace", reschan); err != nil {
			t.Fatal(err)
		}
		doJob()

		if _, err := conn.StopUnit(serviceTargetBase, "replace", reschan); err != nil {
			t.Fatal(err)
		}
		doJob()

		if _, err := conn.StopUnit(proxyContentBase, "replace", reschan); err != nil {
			t.Fatal(err)
		}
		doJob()
	}()

	expected := "HELO\n"
	sockConn, err := net.Dial("tcp", fmt.Sprintf("localhost:%d", port))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := fmt.Fprintf(sockConn, expected); err != nil {
		t.Fatal(err)
	}

	answer, err := bufio.NewReader(sockConn).ReadString('\n')
	if err != nil {
		t.Fatal(err)
	}

	if answer != expected {
		t.Fatalf("Expected %q, Got %q", expected, answer)
	}

	return
}
