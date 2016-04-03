// Copyright 2015 The rkt Authors
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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/coreos/rkt/tests/testutils"
	"github.com/coreos/rkt/tests/testutils/logger"
	"github.com/vishvananda/netlink"
)

/*
 * Host network
 * ---
 * Container must have the same network namespace as the host
 */
func TestNetHost(t *testing.T) {
	testImageArgs := []string{"--exec=/inspect --print-netns"}
	testImage := patchTestACI("rkt-inspect-networking.aci", testImageArgs...)
	defer os.Remove(testImage)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf("%s --net=host --debug --insecure-options=image run --mds-register=false %s", ctx.Cmd(), testImage)
	child := spawnOrFail(t, cmd)
	ctx.RegisterChild(child)
	defer waitOrFail(t, child, 0)

	expectedRegex := `NetNS: (net:\[\d+\])`
	result, out, err := expectRegexWithOutput(child, expectedRegex)
	if err != nil {
		t.Fatalf("Error: %v\nOutput: %v", err, out)
	}

	ns, err := os.Readlink("/proc/self/ns/net")
	if err != nil {
		t.Fatalf("Cannot evaluate NetNS symlink: %v", err)
	}

	if nsChanged := ns != result[1]; nsChanged {
		t.Fatalf("container left host netns")
	}
}

/*
 * Host networking
 * ---
 * Container launches http server which must be reachable by the host via the
 * localhost address
 */
func TestNetHostConnectivity(t *testing.T) {
	logger.SetLogger(t)

	httpPort, err := testutils.GetNextFreePort4()
	if err != nil {
		t.Fatalf("%v", err)
	}
	httpServeAddr := fmt.Sprintf("0.0.0.0:%v", httpPort)
	httpGetAddr := fmt.Sprintf("http://127.0.0.1:%v", httpPort)

	testImageArgs := []string{"--exec=/inspect --serve-http=" + httpServeAddr}
	testImage := patchTestACI("rkt-inspect-networking.aci", testImageArgs...)
	defer os.Remove(testImage)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf("%s --net=host --debug --insecure-options=image run --mds-register=false %s", ctx.Cmd(), testImage)
	child := spawnOrFail(t, cmd)
	ctx.RegisterChild(child)

	ga := testutils.NewGoroutineAssistant(t)
	ga.Add(2)

	// Child opens the server
	go func() {
		defer ga.Done()
		ga.WaitOrFail(child)
	}()

	// Host connects to the child
	go func() {
		defer ga.Done()
		expectedRegex := `serving on`
		_, out, err := expectRegexWithOutput(child, expectedRegex)
		if err != nil {
			ga.Fatalf("Error: %v\nOutput: %v", err, out)
		}
		body, err := testutils.HTTPGet(httpGetAddr)
		if err != nil {
			ga.Fatalf("%v\n", err)
		}
		t.Logf("HTTP-Get received: %s", body)
	}()

	ga.Wait()
}

/*
 * None networking
 * ---
 * must be in an empty netns
 */
func TestNetNone(t *testing.T) {
	testImageArgs := []string{"--exec=/inspect --print-netns --print-iface-count"}
	testImage := patchTestACI("rkt-inspect-networking.aci", testImageArgs...)
	defer os.Remove(testImage)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf("%s --debug --insecure-options=image run --net=none --mds-register=false %s", ctx.Cmd(), testImage)

	child := spawnOrFail(t, cmd)
	defer waitOrFail(t, child, 0)
	expectedRegex := `NetNS: (net:\[\d+\])`
	result, out, err := expectRegexWithOutput(child, expectedRegex)
	if err != nil {
		t.Fatalf("Error: %v\nOutput: %v", err, out)
	}

	ns, err := os.Readlink("/proc/self/ns/net")
	if err != nil {
		t.Fatalf("Cannot evaluate NetNS symlink: %v", err)
	}

	if nsChanged := ns != result[1]; !nsChanged {
		t.Fatalf("container did not leave host netns")
	}

	expectedRegex = `Interface count: (\d+)`
	result, out, err = expectRegexWithOutput(child, expectedRegex)
	if err != nil {
		t.Fatalf("Error: %v\nOutput: %v", err, out)
	}
	ifaceCount, err := strconv.Atoi(result[1])
	if err != nil {
		t.Fatalf("Error parsing interface count: %v\nOutput: %v", err, out)
	}
	if ifaceCount != 1 {
		t.Fatalf("Interface count must be 1 not %q", ifaceCount)
	}
}

/*
 * Default net
 * ---
 * Container must be in a separate network namespace
 */
func TestNetDefaultNetNS(t *testing.T) {
	testImageArgs := []string{"--exec=/inspect --print-netns"}
	testImage := patchTestACI("rkt-inspect-networking.aci", testImageArgs...)
	defer os.Remove(testImage)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	f := func(argument string) {
		cmd := fmt.Sprintf("%s --debug --insecure-options=image run %s --mds-register=false %s", ctx.Cmd(), argument, testImage)
		child := spawnOrFail(t, cmd)
		defer waitOrFail(t, child, 0)

		expectedRegex := `NetNS: (net:\[\d+\])`
		result, out, err := expectRegexWithOutput(child, expectedRegex)
		if err != nil {
			t.Fatalf("Error: %v\nOutput: %v", err, out)
		}

		ns, err := os.Readlink("/proc/self/ns/net")
		if err != nil {
			t.Fatalf("Cannot evaluate NetNS symlink: %v", err)
		}

		if nsChanged := ns != result[1]; !nsChanged {
			t.Fatalf("container did not leave host netns")
		}

	}
	f("--net=default")
	f("")
}

/*
 * Default net
 * ---
 * Host launches http server on all interfaces in the host netns
 * Container must be able to connect via any IP address of the host in the
 * default network, which is NATed
 * TODO: test connection to host on an outside interface
 */
func TestNetDefaultConnectivity(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	f := func(argument string) {
		httpPort, err := testutils.GetNextFreePort4()
		if err != nil {
			t.Fatalf("%v", err)
		}
		httpServeAddr := fmt.Sprintf("0.0.0.0:%v", httpPort)
		httpServeTimeout := 30

		nonLoIPv4, err := testutils.GetNonLoIfaceIPv4()
		if err != nil {
			t.Fatalf("%v", err)
		}
		if nonLoIPv4 == "" {
			t.Skipf("Can not find any NAT'able IPv4 on the host, skipping..")
		}

		httpGetAddr := fmt.Sprintf("http://%v:%v", nonLoIPv4, httpPort)
		t.Log("Telling the child to connect via", httpGetAddr)

		testImageArgs := []string{fmt.Sprintf("--exec=/inspect --get-http=%v", httpGetAddr)}
		testImage := patchTestACI("rkt-inspect-networking.aci", testImageArgs...)
		defer os.Remove(testImage)

		hostname, err := os.Hostname()
		if err != nil {
			t.Fatalf("Error getting hostname: %v", err)
		}

		ga := testutils.NewGoroutineAssistant(t)
		ga.Add(2)

		// Host opens the server
		go func() {
			defer ga.Done()
			err := testutils.HTTPServe(httpServeAddr, httpServeTimeout)
			if err != nil {
				ga.Fatalf("Error during HTTPServe: %v", err)
			}
		}()

		// Child connects to host
		go func() {
			defer ga.Done()
			cmd := fmt.Sprintf("%s --debug --insecure-options=image run %s --mds-register=false %s", ctx.Cmd(), argument, testImage)
			child := ga.SpawnOrFail(cmd)
			defer ga.WaitOrFail(child)

			expectedRegex := `HTTP-Get received: (.*)\r`
			result, out, err := expectRegexWithOutput(child, expectedRegex)
			if err != nil {
				ga.Fatalf("Error: %v\nOutput: %v", err, out)
			}
			if result[1] != hostname {
				ga.Fatalf("Hostname received by client `%v` doesn't match `%v`", result[1], hostname)
			}
		}()

		ga.Wait()
	}
	f("--net=default")
	f("")
}

/*
 * Default-restricted net
 * ---
 * Container launches http server on all its interfaces
 * Host must be able to connects to container's http server via container's
 * eth0's IPv4
 * TODO: verify that the container isn't NATed
 */
func TestNetDefaultRestrictedConnectivity(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	f := func(argument string) {
		httpPort, err := testutils.GetNextFreePort4()
		if err != nil {
			t.Fatalf("%v", err)
		}
		httpServeAddr := fmt.Sprintf("0.0.0.0:%v", httpPort)
		iface := "eth0"

		testImageArgs := []string{fmt.Sprintf("--exec=/inspect --print-ipv4=%v --serve-http=%v", iface, httpServeAddr)}
		testImage := patchTestACI("rkt-inspect-networking.aci", testImageArgs...)
		defer os.Remove(testImage)

		cmd := fmt.Sprintf("%s --debug --insecure-options=image run %s --mds-register=false %s", ctx.Cmd(), argument, testImage)
		child := spawnOrFail(t, cmd)

		expectedRegex := `IPv4: (.*)\r`
		result, out, err := expectRegexWithOutput(child, expectedRegex)
		if err != nil {
			t.Fatalf("Error: %v\nOutput: %v", err, out)
		}
		httpGetAddr := fmt.Sprintf("http://%v:%v", result[1], httpPort)

		ga := testutils.NewGoroutineAssistant(t)
		ga.Add(2)

		// Child opens the server
		go func() {
			defer ga.Done()
			ga.WaitOrFail(child)
		}()

		// Host connects to the child
		go func() {
			defer ga.Done()
			expectedRegex := `serving on`
			_, out, err := expectRegexWithOutput(child, expectedRegex)
			if err != nil {
				ga.Fatalf("Error: %v\nOutput: %v", err, out)
			}
			body, err := testutils.HTTPGet(httpGetAddr)
			if err != nil {
				ga.Fatalf("%v\n", err)
			}
			t.Logf("HTTP-Get received: %s", body)
		}()

		ga.Wait()
	}
	f("--net=default-restricted")
}

/*
 * Default net port forwarding connectivity
 * ---
 * Container launches http server on all its interfaces
 * Host must be able to connect to container's http server on it's own interfaces
 */
func TestNetDefaultPortFwdConnectivity(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	bannedPorts := make(map[int]struct{}, 0)
	f := func(httpGetIP string, rktArg string, shouldSucceed bool) {
		httpPort, err := testutils.GetNextFreePort4Banned(bannedPorts)
		if err != nil {
			t.Fatalf("%v", err)
		}
		bannedPorts[httpPort] = struct{}{}

		httpServeAddr := fmt.Sprintf("0.0.0.0:%d", httpPort)
		testImageArgs := []string{
			fmt.Sprintf("--ports=http,protocol=tcp,port=%d", httpPort),
			fmt.Sprintf("--exec=/inspect --serve-http=%v", httpServeAddr),
		}
		testImage := patchTestACI("rkt-inspect-networking.aci", testImageArgs...)
		defer os.Remove(testImage)

		cmd := fmt.Sprintf(
			"%s --debug --insecure-options=image run --port=http:%d %s --mds-register=false %s",
			ctx.Cmd(), httpPort, rktArg, testImage)
		child := spawnOrFail(t, cmd)

		httpGetAddr := fmt.Sprintf("http://%v:%v", httpGetIP, httpPort)

		ga := testutils.NewGoroutineAssistant(t)
		ga.Add(2)

		// Child opens the server
		go func() {
			defer ga.Done()
			ga.WaitOrFail(child)
		}()

		// Host connects to the child via the forward port on localhost
		go func() {
			defer ga.Done()
			expectedRegex := `serving on`
			_, out, err := expectRegexWithOutput(child, expectedRegex)
			if err != nil {
				ga.Fatalf("Error: %v\nOutput: %v", err, out)
			}
			body, err := testutils.HTTPGet(httpGetAddr)
			switch {
			case err != nil && shouldSucceed:
				ga.Fatalf("%v\n", err)
			case err == nil && !shouldSucceed:
				ga.Fatalf("HTTP-Get to %q should have failed! But received %q", httpGetAddr, body)
			case err != nil && !shouldSucceed:
				child.Close()
				fallthrough
			default:
				t.Logf("HTTP-Get received: %s", body)
			}
		}()

		ga.Wait()
	}
	f("172.16.28.1", "--net=default", true)
	f("127.0.0.1", "--net=default", true)

	// TODO: ensure that default-restricted is not accessible from non-host
	// f("172.16.28.1", "--net=default-restricted", true)
	// f("127.0.0.1", "--net=default-restricted", true)
}

func writeNetwork(t *testing.T, net networkTemplateT, netd string) error {
	var err error
	path := filepath.Join(netd, net.Name+".conf")
	file, err := os.Create(path)
	if err != nil {
		t.Errorf("%v", err)
	}

	b, err := json.Marshal(net)
	if err != nil {
		return err
	}

	fmt.Println("Writing", net.Name, "to", path)
	_, err = file.Write(b)
	if err != nil {
		return err
	}

	return nil
}

type networkTemplateT struct {
	Name      string
	Type      string
	Master    string `json:"master,omitempty"`
	IpMasq    bool
	IsGateway bool
	Ipam      ipamTemplateT
}

type ipamTemplateT struct {
	Type   string
	Subnet string              `json:"subnet,omitempty"`
	Routes []map[string]string `json:"routes,omitempty"`
}

func TestNetTemplates(t *testing.T) {
	net := networkTemplateT{
		Name: "ptp0",
		Type: "ptp",
		Ipam: ipamTemplateT{
			Type:   "host-local",
			Subnet: "11.11.3.0/24",
			Routes: []map[string]string{{"dst": "0.0.0.0/0"}},
		},
	}

	b, err := json.Marshal(net)
	if err != nil {
		t.Fatalf("%v", err)
	}
	expected := `{"Name":"ptp0","Type":"ptp","IpMasq":false,"IsGateway":false,"Ipam":{"Type":"host-local","subnet":"11.11.3.0/24","routes":[{"dst":"0.0.0.0/0"}]}}`
	if string(b) != expected {
		t.Fatalf("Template extected:\n%v\ngot:\n%v\n", expected, string(b))
	}
}

func prepareTestNet(t *testing.T, ctx *testutils.RktRunCtx, nt networkTemplateT) (netdir string) {
	configdir := ctx.LocalDir()
	netdir = filepath.Join(configdir, "net.d")
	err := os.MkdirAll(netdir, 0644)
	if err != nil {
		t.Fatalf("Cannot create netdir: %v", err)
	}
	err = writeNetwork(t, nt, netdir)
	if err != nil {
		t.Fatalf("Cannot write network file: %v", err)
	}
	return netdir
}

/*
 * Two containers spawn in the same custom network.
 * ---
 * Container 1 opens the http server
 * Container 2 fires a HTTPGet on it
 * The body of the HTTPGet is Container 1's hostname, which must match
 */
func testNetCustomDual(t *testing.T, nt networkTemplateT) {
	httpPort, err := testutils.GetNextFreePort4()
	if err != nil {
		t.Fatalf("%v", err)
	}

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	netdir := prepareTestNet(t, ctx, nt)
	defer os.RemoveAll(netdir)

	container1IPv4, container1Hostname := make(chan string), make(chan string)
	ga := testutils.NewGoroutineAssistant(t)
	ga.Add(2)

	go func() {
		defer ga.Done()
		httpServeAddr := fmt.Sprintf("0.0.0.0:%v", httpPort)
		testImageArgs := []string{"--exec=/inspect --print-ipv4=eth0 --serve-http=" + httpServeAddr}
		testImage := patchTestACI("rkt-inspect-networking1.aci", testImageArgs...)
		defer os.Remove(testImage)

		cmd := fmt.Sprintf("%s --debug --insecure-options=image run --net=%v --mds-register=false %s", ctx.Cmd(), nt.Name, testImage)
		child := ga.SpawnOrFail(cmd)
		defer ga.WaitOrFail(child)

		expectedRegex := `IPv4: (\d+\.\d+\.\d+\.\d+)`
		result, out, err := expectRegexTimeoutWithOutput(child, expectedRegex, 30*time.Second)
		if err != nil {
			ga.Fatalf("Error: %v\nOutput: %v", err, out)
		}
		container1IPv4 <- result[1]
		expectedRegex = `(rkt-.*): serving on`
		result, out, err = expectRegexTimeoutWithOutput(child, expectedRegex, 30*time.Second)
		if err != nil {
			ga.Fatalf("Error: %v\nOutput: %v", err, out)
		}
		container1Hostname <- result[1]
	}()

	go func() {
		defer ga.Done()

		var httpGetAddr string
		httpGetAddr = fmt.Sprintf("http://%v:%v", <-container1IPv4, httpPort)

		testImageArgs := []string{"--exec=/inspect --get-http=" + httpGetAddr}
		testImage := patchTestACI("rkt-inspect-networking2.aci", testImageArgs...)
		defer os.Remove(testImage)

		cmd := fmt.Sprintf("%s --debug --insecure-options=image run --net=%v --mds-register=false %s", ctx.Cmd(), nt.Name, testImage)
		child := ga.SpawnOrFail(cmd)
		defer ga.WaitOrFail(child)

		expectedHostname := <-container1Hostname
		expectedRegex := `HTTP-Get received: (.*)\r`
		result, out, err := expectRegexTimeoutWithOutput(child, expectedRegex, 20*time.Second)
		if err != nil {
			ga.Fatalf("Error: %v\nOutput: %v", err, out)
		}
		t.Logf("HTTP-Get received: %s", result[1])
		receivedHostname := result[1]

		if receivedHostname != expectedHostname {
			ga.Fatalf("Received hostname `%v` doesn't match `%v`", receivedHostname, expectedHostname)
		}
	}()

	ga.Wait()
}

/*
 * Host launches http server on all interfaces in the host netns
 * Container must be able to connect via any IP address of the host in the
 * macvlan network, which is NAT
 * TODO: test connection to host on an outside interface
 */
func testNetCustomNatConnectivity(t *testing.T, nt networkTemplateT) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	netdir := prepareTestNet(t, ctx, nt)
	defer os.RemoveAll(netdir)

	httpPort, err := testutils.GetNextFreePort4()
	if err != nil {
		t.Fatalf("%v", err)
	}
	httpServeAddr := fmt.Sprintf("0.0.0.0:%v", httpPort)
	httpServeTimeout := 30

	nonLoIPv4, err := testutils.GetNonLoIfaceIPv4()
	if err != nil {
		t.Fatalf("%v", err)
	}
	if nonLoIPv4 == "" {
		t.Skipf("Can not find any NAT'able IPv4 on the host, skipping..")
	}

	httpGetAddr := fmt.Sprintf("http://%v:%v", nonLoIPv4, httpPort)
	t.Log("Telling the child to connect via", httpGetAddr)

	ga := testutils.NewGoroutineAssistant(t)
	ga.Add(2)

	// Host opens the server
	go func() {
		defer ga.Done()
		err := testutils.HTTPServe(httpServeAddr, httpServeTimeout)
		if err != nil {
			ga.Fatalf("Error during HTTPServe: %v", err)
		}
	}()

	// Child connects to host
	hostname, err := os.Hostname()
	go func() {
		defer ga.Done()
		testImageArgs := []string{fmt.Sprintf("--exec=/inspect --get-http=%v", httpGetAddr)}
		testImage := patchTestACI("rkt-inspect-networking.aci", testImageArgs...)
		defer os.Remove(testImage)

		cmd := fmt.Sprintf("%s --debug --insecure-options=image run --net=%v --mds-register=false %s", ctx.Cmd(), nt.Name, testImage)
		child := ga.SpawnOrFail(cmd)
		defer ga.WaitOrFail(child)

		expectedRegex := `HTTP-Get received: (.*)\r`
		result, out, err := expectRegexWithOutput(child, expectedRegex)
		if err != nil {
			ga.Fatalf("Error: %v\nOutput: %v", err, out)
		}
		if result[1] != hostname {
			ga.Fatalf("Hostname received by client `%v` doesn't match `%v`", result[1], hostname)
		}
	}()

	ga.Wait()
}

func TestNetCustomPtp(t *testing.T) {
	nt := networkTemplateT{
		Name:   "ptp0",
		Type:   "ptp",
		IpMasq: true,
		Ipam: ipamTemplateT{
			Type:   "host-local",
			Subnet: "11.11.1.0/24",
			Routes: []map[string]string{
				{"dst": "0.0.0.0/0"},
			},
		},
	}
	testNetCustomNatConnectivity(t, nt)
	testNetCustomDual(t, nt)
}

func TestNetCustomMacvlan(t *testing.T) {
	iface, _, err := testutils.GetNonLoIfaceWithAddrs(netlink.FAMILY_V4)
	if err != nil {
		t.Fatalf("Error while getting non-lo host interface: %v\n", err)
	}
	if iface.Name == "" {
		t.Skipf("Cannot run test without non-lo host interface")
	}

	nt := networkTemplateT{
		Name:   "macvlan0",
		Type:   "macvlan",
		Master: iface.Name,
		Ipam: ipamTemplateT{
			Type:   "host-local",
			Subnet: "11.11.2.0/24",
		},
	}
	testNetCustomDual(t, nt)
}

func TestNetCustomBridge(t *testing.T) {
	iface, _, err := testutils.GetNonLoIfaceWithAddrs(netlink.FAMILY_V4)
	if err != nil {
		t.Fatalf("Error while getting non-lo host interface: %v\n", err)
	}
	if iface.Name == "" {
		t.Skipf("Cannot run test without non-lo host interface")
	}

	nt := networkTemplateT{
		Name:      "bridge0",
		Type:      "bridge",
		IpMasq:    true,
		IsGateway: true,
		Master:    iface.Name,
		Ipam: ipamTemplateT{
			Type:   "host-local",
			Subnet: "11.11.3.0/24",
			Routes: []map[string]string{
				{"dst": "0.0.0.0/0"},
			},
		},
	}
	testNetCustomNatConnectivity(t, nt)
	testNetCustomDual(t, nt)
}

func TestNetOverride(t *testing.T) {
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
		Name:   "overridemacvlan",
		Type:   "macvlan",
		Master: iface.Name,
		Ipam: ipamTemplateT{
			Type:   "host-local",
			Subnet: "11.11.4.0/24",
		},
	}

	netdir := prepareTestNet(t, ctx, nt)
	defer os.RemoveAll(netdir)

	testImageArgs := []string{"--exec=/inspect --print-ipv4=eth0"}
	testImage := patchTestACI("rkt-inspect-networking1.aci", testImageArgs...)
	defer os.Remove(testImage)

	expectedIP := "11.11.4.244"

	cmd := fmt.Sprintf("%s --debug --insecure-options=image run --net=all --net=\"%s:IP=%s\" --mds-register=false %s", ctx.Cmd(), nt.Name, expectedIP, testImage)
	child := spawnOrFail(t, cmd)
	defer waitOrFail(t, child, 0)

	expectedRegex := `IPv4: (\d+\.\d+\.\d+\.\d+)`
	result, out, err := expectRegexTimeoutWithOutput(child, expectedRegex, 30*time.Second)
	if err != nil {
		t.Fatalf("Error: %v\nOutput: %v", err, out)
		return
	}

	containerIP := result[1]
	if expectedIP != containerIP {
		t.Fatalf("overriding IP did not work: Got %q but expected %q", containerIP, expectedIP)
	}
}
