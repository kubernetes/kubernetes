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

// +build host coreos src kvm

package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/coreos/rkt/networking/netinfo"
	"github.com/coreos/rkt/pkg/fileutil"
	"github.com/coreos/rkt/tests/testutils"
	"github.com/coreos/rkt/tests/testutils/logger"
	"github.com/vishvananda/netlink"
)

/*
 * Host network
 * ---
 * Container must have the same network namespace as the host
 */
func NewNetHostTest() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
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
	})
}

/*
 * Host networking
 * ---
 * Container launches http server which must be reachable by the host via the
 * localhost address
 */
func NewNetHostConnectivityTest() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
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
	})
}

/*
 * None networking
 * ---
 * must be in an empty netns
 */
func NewNetNoneTest() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
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
	})
}

/*
 * Default net
 * ---
 * Container must be in a separate network namespace
 */
func NewTestNetDefaultNetNS() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
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
	})
}

/*
 * Default net
 * ---
 * Host launches http server on all interfaces in the host netns
 * Container must be able to connect via any IP address of the host in the
 * default network, which is NATed
 * TODO: test connection to host on an outside interface
 */
func NewNetDefaultConnectivityTest() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
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
	})
}

/*
 * Default-restricted net
 * ---
 * Container launches http server on all its interfaces
 * Host must be able to connects to container's http server via container's
 * eth0's IPv4
 * TODO: verify that the container isn't NATed
 */
func NewTestNetDefaultRestrictedConnectivity() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
		ctx := testutils.NewRktRunCtx()
		defer ctx.Cleanup()

		f := func(argument string) {
			httpPort := "8080"
			iface := "eth0"

			testImageArgs := []string{fmt.Sprintf("--exec=/inspect --print-ipv4=%v --serve-http=0.0.0.0:%v", iface, httpPort)}
			testImage := patchTestACI("rkt-inspect-networking.aci", testImageArgs...)
			defer os.Remove(testImage)

			cmd := fmt.Sprintf("%s --insecure-options=image run %s --mds-register=false %s", ctx.Cmd(), argument, testImage)
			child := spawnOrFail(t, cmd)

			// Wait for the container to print out the IP address
			expectedRegex := `IPv4: (\d+\.\d+\.\d+\.\d+)`
			result, out, err := expectRegexWithOutput(child, expectedRegex)
			if err != nil {
				t.Fatalf("Error: %v\nOutput: %v", err, out)
			}
			httpGetAddr := fmt.Sprintf("http://%v:%v", result[1], httpPort)

			// Wait for the container to open the port
			expectedRegex = `serving on`
			_, out, err = expectRegexWithOutput(child, expectedRegex)
			if err != nil {
				t.Fatalf("Error: %v\nOutput: %v", err, out)
			}
			body, err := testutils.HTTPGet(httpGetAddr)
			if err != nil {
				t.Fatalf("%v\n", err)
			}
			t.Logf("HTTP-Get received: %s", body)
			waitOrFail(t, child, 0)
		}
		f("--net=default-restricted")
	})
}

type PortFwdCase struct {
	HttpGetIP     string
	HttpServePort int
	ListenAddress string
	RktArg        string
	ShouldSucceed bool
}

var (
	bannedPorts = make(map[int]struct{}, 0)

	defaultSamePortFwdCase       = PortFwdCase{"172.16.28.1", 0, "", "--net=default", true}
	defaultDiffPortFwdCase       = PortFwdCase{"172.16.28.1", 1024, "", "--net=default", true}
	defaultSpecificIPFwdCase     = PortFwdCase{"172.16.28.1", 1024, "172.16.28.1:", "--net=default", true}
	defaultSpecificIPFwdFailCase = PortFwdCase{"127.0.0.1", 1024, "172.16.28.1:", "--net=default", false}
	defaultLoSamePortFwdCase     = PortFwdCase{"127.0.0.1", 0, "", "--net=default", true}
	defaultLoDiffPortFwdCase     = PortFwdCase{"127.0.0.1", 1014, "", "--net=default", true}

	portFwdBridge = networkTemplateT{
		Name:      "bridge1",
		Type:      "bridge",
		Bridge:    "bridge1",
		IpMasq:    true,
		IsGateway: true,
		Ipam: &ipamTemplateT{
			Type:   "host-local",
			Subnet: "11.11.5.0/24",
			Routes: []map[string]string{
				{"dst": "0.0.0.0/0"},
			},
		},
	}
	bridgeSamePortFwdCase   = PortFwdCase{"11.11.5.1", 0, "", "--net=" + portFwdBridge.Name, true}
	bridgeDiffPortFwdCase   = PortFwdCase{"11.11.5.1", 1024, "", "--net=" + portFwdBridge.Name, true}
	bridgeLoSamePortFwdCase = PortFwdCase{"127.0.0.1", 0, "", "--net=" + portFwdBridge.Name, true}
	bridgeLoDiffPortFwdCase = PortFwdCase{"127.0.0.1", 1024, "", "--net=" + portFwdBridge.Name, true}
)

func (ct PortFwdCase) Execute(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	prepareTestNet(t, ctx, portFwdBridge)

	httpPort, err := testutils.GetNextFreePort4Banned(bannedPorts)
	if err != nil {
		t.Fatalf("%v", err)
	}
	bannedPorts[httpPort] = struct{}{}

	httpServePort := ct.HttpServePort
	if httpServePort == 0 {
		httpServePort = httpPort
	}

	httpServeAddr := fmt.Sprintf("0.0.0.0:%d", httpServePort)
	testImageArgs := []string{
		fmt.Sprintf("--ports=http,protocol=tcp,port=%d", httpServePort),
		fmt.Sprintf("--exec=/inspect --serve-http=%v", httpServeAddr),
	}
	t.Logf("testImageArgs: %v", testImageArgs)
	testImage := patchTestACI("rkt-inspect-networking.aci", testImageArgs...)
	defer os.Remove(testImage)

	cmd := fmt.Sprintf(
		"%s --debug --insecure-options=image run --port=http:%s%d %s --mds-register=false %s",
		ctx.Cmd(), ct.ListenAddress, httpPort, ct.RktArg, testImage)
	child := spawnOrFail(t, cmd)

	httpGetAddr := fmt.Sprintf("http://%v:%v", ct.HttpGetIP, httpPort)

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
		case err != nil && ct.ShouldSucceed:
			ga.Fatalf("%v\n", err)
		case err == nil && !ct.ShouldSucceed:
			ga.Fatalf("HTTP-Get to %q should have failed! But received %q", httpGetAddr, body)
		case err != nil && !ct.ShouldSucceed:
			t.Logf("HTTP-Get failed, as expected: %v", err)
		default:
			t.Logf("HTTP-Get received: %s", body)
		}
	}()

	ga.Wait()
}

type portFwdTest []PortFwdCase

func (ct portFwdTest) Execute(t *testing.T) {
	for _, testCase := range ct {
		testCase.Execute(t)
	}
}

/*
 * Net port forwarding connectivity
 * ---
 * Container launches http server on all its interfaces
 * Host must be able to connect to container's http server on it's own interfaces
 */
func NewNetPortFwdConnectivityTest(cases ...PortFwdCase) testutils.Test {
	return portFwdTest(cases)
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

// Compute what we should pass to the --net parameter at runtime
func (nt *networkTemplateT) NetParameter() string {
	out := nt.Name

	if len(nt.Args) > 0 {
		out += ":" + strings.Join(nt.Args, ";")
	}

	return out
}

type networkTemplateT struct {
	Name       string
	Type       string
	SubnetFile string `json:"subnetFile,omitempty"`
	Master     string `json:"master,omitempty"`
	IpMasq     bool
	IsGateway  bool
	Bridge     string             `json:"bridge,omitempty"`
	Ipam       *ipamTemplateT     `json:",omitempty"`
	Delegate   *delegateTemplateT `json:",omitempty"`
	Args       []string           `json:"args,omitempty"` // Arguments to the CNI plugin, array of "foo=bar" pairs
}

type ipamTemplateT struct {
	Type   string              `json:",omitempty"`
	Subnet string              `json:"subnet,omitempty"`
	Routes []map[string]string `json:"routes,omitempty"`
}

type delegateTemplateT struct {
	Bridge           string `json:"bridge,omitempty"`
	IsDefaultGateway bool   `json:"isDefaultGateway"`
}

func TestNetTemplates(t *testing.T) {
	net := networkTemplateT{
		Name: "ptp0",
		Type: "ptp",
		Args: []string{"two=three", "black=white"},
		Ipam: &ipamTemplateT{
			Type:   "host-local",
			Subnet: "11.11.3.0/24",
			Routes: []map[string]string{{"dst": "0.0.0.0/0"}},
		},
	}

	b, err := json.Marshal(net)
	if err != nil {
		t.Fatalf("%v", err)
	}
	expected := `{"Name":"ptp0","Type":"ptp","IpMasq":false,"IsGateway":false,"Ipam":{"Type":"host-local","subnet":"11.11.3.0/24","routes":[{"dst":"0.0.0.0/0"}]},"args":["two=three","black=white"]}`
	if string(b) != expected {
		t.Fatalf("Template extected:\n%v\ngot:\n%v\n", expected, string(b))
	}
}

// The format of the logfile from cniproxy
type cniProxyResult struct {
	PluginPath string            `json:"pluginPath"`
	Stdin      string            `json:"stdin"`
	Stdout     string            `json:"stdout"`
	Stderr     string            `json:"stderr"`
	ExitCode   int               `json:"exitCode"`
	Env        []string          `json:"env"`
	EnvMap     map[string]string `json:"-"`
}

func parseCNIProxyLog(filepath string) (*cniProxyResult, error) {
	fp, err := os.Open(filepath)
	defer fp.Close()
	if err != nil {
		return nil, err
	}
	res := new(cniProxyResult)
	err = json.NewDecoder(fp).Decode(res)
	if err != nil {
		return nil, err
	}
	res.EnvMap = make(map[string]string)

	for _, envstr := range res.Env {
		vals := strings.SplitN(envstr, "=", 2)
		if len(vals) != 2 {
			continue
		}
		res.EnvMap[vals[0]] = vals[1]
	}

	return res, nil
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

	// If we're proxying the CNI call, then make sure it's in the netdir
	if nt.Type == "cniproxy" {
		dest := filepath.Join(netdir, "cniproxy")
		err := fileutil.CopyRegularFile(testutils.GetValueFromEnvOrPanic("RKT_CNI_PROXY"), dest)
		if err != nil {
			t.Fatalf("Cannot copy cniproxy")
		}
		os.Chmod(dest, 0755)
		if err != nil {
			t.Fatalf("Cannot chmod cniproxy")
		}
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

	prepareTestNet(t, ctx, nt)

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
		expectedRegex = ` ([a-zA-Z0-9\-]*): serving on`
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
		expectedRegex := `HTTP-Get received: (.*?)\r`
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

	prepareTestNet(t, ctx, nt)

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
	if err != nil {
		panic(err)
	}

	go func() {
		defer ga.Done()
		testImageArgs := []string{fmt.Sprintf("--exec=/inspect --get-http=%v", httpGetAddr)}
		testImage := patchTestACI("rkt-inspect-networking.aci", testImageArgs...)
		defer os.Remove(testImage)

		cmd := fmt.Sprintf("%s --debug --insecure-options=image run --net=%v --mds-register=false %s", ctx.Cmd(), nt.Name, testImage)
		child := ga.SpawnOrFail(cmd)
		defer ga.WaitOrFail(child)

		expectedRegex := `HTTP-Get received: (.*?)\r`
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

//Test that the CNI execution environment matches the spec
func NewNetCNIEnvTest() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
		ctx := testutils.NewRktRunCtx()
		defer ctx.Cleanup()

		iface, _, err := testutils.GetNonLoIfaceWithAddrs(netlink.FAMILY_V4)
		if err != nil {
			t.Fatalf("Error while getting non-lo host interface: %v\n", err)
		}
		if iface.Name == "" {
			t.Skipf("Cannot run test without non-lo host interface")
		}

		// Declares a network of type cniproxy, which will record state and
		// proxy through to $X_REAL_PLUGIN
		nt := networkTemplateT{
			Name:      "bridge0",
			Type:      "cniproxy",
			Args:      []string{"X_LOG=output.json", "X_REAL_PLUGIN=bridge"},
			IpMasq:    true,
			IsGateway: true,
			Master:    iface.Name,
			Ipam: &ipamTemplateT{
				Type:   "host-local",
				Subnet: "11.11.3.0/24",
				Routes: []map[string]string{
					{"dst": "0.0.0.0/0"},
				},
			},
		}

		// bring the networking up, copy the proxy
		netdir := prepareTestNet(t, ctx, nt)

		appCmd := "--exec=/inspect -- --print-defaultgwv4 "
		cmd := fmt.Sprintf("%s --debug --insecure-options=image run --net=%v --mds-register=false %s %s",
			ctx.Cmd(), nt.NetParameter(), getInspectImagePath(), appCmd)
		child := spawnOrFail(t, cmd)

		expectedRegex := "DefaultGWv4: 11.11.3.1"

		_, out, err := expectRegexTimeoutWithOutput(child, expectedRegex, 30*time.Second)
		if err != nil {
			t.Fatalf("Error: %v\nOutput: %v", err, out)
		}
		waitOrFail(t, child, 0)

		// Parse the log file
		cniLogFilename := filepath.Join(netdir, "output.json")
		proxyLog, err := parseCNIProxyLog(cniLogFilename)
		if err != nil {
			t.Fatal("Failed to read cniproxy ADD log", err)
		}
		os.Remove(cniLogFilename)

		// Check that the stdin matches the network config file
		expectedConfig, err := ioutil.ReadFile(filepath.Join(netdir, nt.Name+".conf"))
		if err != nil {
			t.Fatal("Failed to read network configuration", err)
		}

		if string(expectedConfig) != proxyLog.Stdin {
			t.Fatalf("CNI plugin stdin incorrect, expected <<%v>>, actual <<%v>>", expectedConfig, proxyLog.Stdin)
		}

		// compare the CNI env against a set of regexes
		checkEnv := func(step string, expectedEnv, actualEnv map[string]string) {
			for k, v := range expectedEnv {
				actual, exists := actualEnv[k]
				if !exists {
					t.Fatalf("Step %s, expected proxy CNI arg %s but not found", step, k)
				}

				re, err := regexp.Compile(v)
				if err != nil {
					t.Fatalf("Step %s, invalid CNI env regex for key %s %v", step, k, err)
				}
				found := re.FindString(actual)
				if found == "" {
					t.Fatalf("step %s cni environment %s was %s but expected pattern %s", step, k, actual, v)
				}
			}
		}

		expectedEnv := map[string]string{
			"CNI_VERSION":     `^0\.1\.0$`,
			"CNI_COMMAND":     `^ADD$`,
			"CNI_IFNAME":      `^eth\d$`,
			"CNI_PATH":        "^" + netdir + ":/usr/lib/rkt/plugins/net:stage1/rootfs/usr/lib/rkt/plugins/net$",
			"CNI_NETNS":       `^/var/run/netns/cni-`,
			"CNI_CONTAINERID": `^[a-fA-F0-9-]{36}$`, //UUID, close enough
		}
		checkEnv("add", expectedEnv, proxyLog.EnvMap)

		/*
			Run rkt GC, ensure the CNI invocation looks sane
		*/
		ctx.RunGC()
		proxyLog, err = parseCNIProxyLog(cniLogFilename)
		if err != nil {
			t.Fatal("Failed to read cniproxy DEL log", err)
		}
		os.Remove(cniLogFilename)

		expectedEnv["CNI_COMMAND"] = `^DEL$`
		checkEnv("del", expectedEnv, proxyLog.EnvMap)

	})
}

// Test that CNI invocations which return DNS information are carried through to /etc/resolv.conf
func NewNetCNIDNSTest() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
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
			Name:      "bridge0",
			Type:      "cniproxy",
			Args:      []string{"X_LOG=output.json", "X_REAL_PLUGIN=bridge", "X_ADD_DNS=1"},
			IpMasq:    true,
			IsGateway: true,
			Master:    iface.Name,
			Ipam: &ipamTemplateT{
				Type:   "host-local",
				Subnet: "11.11.3.0/24",
				Routes: []map[string]string{
					{"dst": "0.0.0.0/0"},
				},
			},
		}

		// bring the networking up, copy the proxy
		prepareTestNet(t, ctx, nt)

		ga := testutils.NewGoroutineAssistant(t)
		ga.Add(1)

		go func() {
			defer ga.Done()

			appCmd := "--exec=/inspect -- --read-file --file-name=/etc/resolv.conf"
			cmd := fmt.Sprintf("%s --debug --insecure-options=image run --net=%v --mds-register=false %s %s",
				ctx.Cmd(), nt.NetParameter(), getInspectImagePath(), appCmd)
			child := ga.SpawnOrFail(cmd)
			defer ga.WaitOrFail(child)

			expectedRegex := "nameserver 1.2.3.4"

			_, out, err := expectRegexTimeoutWithOutput(child, expectedRegex, 30*time.Second)
			if err != nil {
				ga.Fatalf("Error: %v\nOutput: %v", err, out)
			}
		}()

		ga.Wait()
	})
}

// Test that `rkt run --dns` overrides CNI DNS
func NewNetCNIDNSArgTest() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
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
			Name:      "bridge0",
			Type:      "cniproxy",
			Args:      []string{"X_LOG=output.json", "X_REAL_PLUGIN=bridge", "X_ADD_DNS=1"},
			IpMasq:    true,
			IsGateway: true,
			Master:    iface.Name,
			Ipam: &ipamTemplateT{
				Type:   "host-local",
				Subnet: "11.11.3.0/24",
				Routes: []map[string]string{
					{"dst": "0.0.0.0/0"},
				},
			},
		}

		// bring the networking up, copy the proxy
		prepareTestNet(t, ctx, nt)

		appCmd := "--exec=/inspect -- --read-file --file-name=/etc/resolv.conf"
		cmd := fmt.Sprintf("%s --debug --insecure-options=image run --net=%v --mds-register=false --dns=244.244.244.244 %s %s",
			ctx.Cmd(), nt.NetParameter(), getInspectImagePath(), appCmd)
		child := spawnOrFail(t, cmd)
		defer waitOrFail(t, child, 0)

		expectedRegex := "nameserver 244.244.244.244"

		_, out, err := expectRegexTimeoutWithOutput(child, expectedRegex, 30*time.Second)
		if err != nil {
			t.Fatalf("Error: %v\nOutput: %v", err, out)
		}
	})
}

// Test that `rkt run --dns=none` means no resolv.conf is created, even when
// CNI returns DNS informationparseHostsEntries(flagHosts)
func NewNetCNIDNSArgNoneTest() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
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
			Name:      "bridge0",
			Type:      "cniproxy",
			Args:      []string{"X_LOG=output.json", "X_REAL_PLUGIN=bridge", "X_ADD_DNS=1"},
			IpMasq:    true,
			IsGateway: true,
			Master:    iface.Name,
			Ipam: &ipamTemplateT{
				Type:   "host-local",
				Subnet: "11.11.3.0/24",
				Routes: []map[string]string{
					{"dst": "0.0.0.0/0"},
				},
			},
		}

		// bring the networking up, copy the proxy
		prepareTestNet(t, ctx, nt)

		appCmd := "--exec=/inspect -- --stat-file --file-name=/etc/resolv.conf"
		cmd := fmt.Sprintf("%s --debug --insecure-options=image run --net=%v --mds-register=false --dns=none %s %s",
			ctx.Cmd(), nt.NetParameter(), getInspectImagePath(), appCmd)
		child := spawnOrFail(t, cmd)
		ctx.RegisterChild(child)
		defer waitOrFail(t, child, 254)

		expectedRegex := `Cannot stat file "/etc/resolv.conf": stat /etc/resolv.conf: no such file or directory`

		_, out, err := expectRegexTimeoutWithOutput(child, expectedRegex, 30*time.Second)
		if err != nil {
			t.Fatalf("Error: %v\nOutput: %v", err, out)
		}
	})
}

func NewNetCustomPtpTest(runCustomDual bool) testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
		nt := networkTemplateT{
			Name:   "ptp0",
			Type:   "ptp",
			IpMasq: true,
			Ipam: &ipamTemplateT{
				Type:   "host-local",
				Subnet: "11.11.1.0/24",
				Routes: []map[string]string{
					{"dst": "0.0.0.0/0"},
				},
			},
		}
		testNetCustomNatConnectivity(t, nt)
		if runCustomDual {
			testNetCustomDual(t, nt)
		}
	})
}

func NewNetCustomMacvlanTest() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
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
			Ipam: &ipamTemplateT{
				Type:   "host-local",
				Subnet: "11.11.2.0/24",
			},
		}
		testNetCustomDual(t, nt)
	})
}

func NewNetCustomBridgeTest() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
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
			Ipam: &ipamTemplateT{
				Type:   "host-local",
				Subnet: "11.11.3.0/24",
				Routes: []map[string]string{
					{"dst": "0.0.0.0/0"},
				},
			},
		}
		testNetCustomNatConnectivity(t, nt)
		testNetCustomDual(t, nt)
	})
}

func NewNetOverrideTest() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
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
			Ipam: &ipamTemplateT{
				Type:   "host-local",
				Subnet: "11.11.4.0/24",
			},
		}

		prepareTestNet(t, ctx, nt)

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
	})
}

/*
 * Pass the IP arg to the default networks, ensure it works
 */
func NewNetDefaultIPArgTest() testutils.Test {
	doTest := func(netArg, expectedIP string, t *testing.T) {
		ctx := testutils.NewRktRunCtx()
		defer ctx.Cleanup()

		appCmd := "--exec=/inspect -- --print-ipv4=eth0"
		cmd := fmt.Sprintf("%s --debug --insecure-options=image run --net=\"%s\" --mds-register=false %s %s",
			ctx.Cmd(), netArg, getInspectImagePath(), appCmd)
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
			t.Fatalf("--net=%s setting IP failed: Got %q but expected %q", netArg, containerIP, expectedIP)
		}
	}
	return testutils.TestFunc(func(t *testing.T) {
		doTest("default:IP=172.16.28.123", "172.16.28.123", t)
		doTest("default-restricted:IP=172.17.42.42", "172.17.42.42", t)
	})
}

/*
 * Try and start two containers with the same IP, ensure
 * the second invocation fails
 */
func NewNetIPConflictTest() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
		ctx := testutils.NewRktRunCtx()
		defer ctx.Cleanup()

		// Launch one container and grab the IP it uses -- and have it idle
		appCmd := "--exec=/inspect -- --print-ipv4=eth0  --serve-http=0.0.0.0:80"
		cmd1 := fmt.Sprintf("%s --debug --insecure-options=image run --mds-register=false %s %s",
			ctx.Cmd(), getInspectImagePath(), appCmd)

		child1 := spawnOrFail(t, cmd1)

		expectedRegex := `IPv4: (\d+\.\d+\.\d+\.\d+)`
		result, out, err := expectRegexTimeoutWithOutput(child1, expectedRegex, 30*time.Second)
		if err != nil {
			t.Fatalf("Error: %v\nOutput: %v", err, out)
			return
		}
		ip := result[1]

		// Launch a second container with the same IP
		cmd2 := fmt.Sprintf("%s --debug --insecure-options=image run --net=\"default:IP=%s\" %s --exec=/inspect -- --print-ipv4=eth0",
			ctx.Cmd(), ip, getInspectImagePath())
		child2 := spawnOrFail(t, cmd2)

		expectedOutput := fmt.Sprintf(`requested IP address "%s" is not available in network: default`, ip)

		_, out, err = expectRegexTimeoutWithOutput(child2, expectedOutput, 10*time.Second)
		if err != nil {
			t.Fatalf("Error: %v\nOutput: %v", err, out)
			return
		}

		// Clean up
		waitOrFail(t, child2, 254)
		syscall.Kill(child1.Cmd.Process.Pid, syscall.SIGTERM)
		waitOrFail(t, child1, 0)
	})
}

func NewTestNetLongName() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
		nt := networkTemplateT{
			Name:   "thisnameiswaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaytoolong",
			Type:   "ptp",
			IpMasq: true,
			Ipam: &ipamTemplateT{
				Type:   "host-local",
				Subnet: "11.11.6.0/24",
				Routes: []map[string]string{
					{"dst": "0.0.0.0/0"},
				},
			},
		}
		testNetCustomNatConnectivity(t, nt)
	})
}

/*
 * mockFlannelNetwork creates fake flannel network status file and configuration pointing to this network.
 * We won't have connectivity, but we could check if: netName was correct and if default gateway was set.
 */
func mockFlannelNetwork(t *testing.T, ctx *testutils.RktRunCtx) (string, networkTemplateT, error) {
	// write fake flannel info
	subnetPath := filepath.Join(ctx.DataDir(), "subnet.env")
	file, err := os.Create(subnetPath)
	if err != nil {
		return "", networkTemplateT{}, err
	}
	mockedFlannel := strings.Join([]string{
		"FLANNEL_NETWORK=11.11.0.0/16",
		"FLANNEL_SUBNET=11.11.3.1/24",
		"FLANNEL_MTU=1472",
		"FLANNEL_IPMASQ=true",
	}, "\n")
	if _, err = file.WriteString(mockedFlannel); err != nil {
		return "", networkTemplateT{}, err
	}

	file.Close()

	// write net config for "flannel" based network
	ntFlannel := networkTemplateT{
		Name:       "rkt.kubernetes.io",
		Type:       "flannel",
		SubnetFile: subnetPath,
		Delegate: &delegateTemplateT{
			IsDefaultGateway: true,
		},
	}

	netdir := prepareTestNet(t, ctx, ntFlannel)

	return netdir, ntFlannel, nil
}

/*
 * NewNetPreserveNetNameTest checks if netName is set if network is configured via flannel
 */
func NewNetPreserveNetNameTest() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
		ctx := testutils.NewRktRunCtx()
		defer ctx.Cleanup()

		_, ntFlannel, err := mockFlannelNetwork(t, ctx)
		if err != nil {
			t.Errorf("Can't mock flannel network: %v", err)
		}

		defer os.Remove(ntFlannel.SubnetFile)

		podUUIDFile := filepath.Join(ctx.DataDir(), "pod_uuid")
		defer os.Remove(podUUIDFile)

		// start container with 'flannel' network
		testImageArgs := []string{"--exec=/inspect"}
		testImage := patchTestACI("rkt-inspect-networking.aci", testImageArgs...)
		defer os.Remove(testImage)

		cmd := fmt.Sprintf("%s --debug --insecure-options=image run --uuid-file-save=%s --net=%s --mds-register=false %s", ctx.Cmd(), podUUIDFile, ntFlannel.Name, testImage)
		spawnAndWaitOrFail(t, cmd, 0)

		podUUID, err := ioutil.ReadFile(podUUIDFile)
		if err != nil {
			t.Fatalf("Can't read pod UUID: %v", err)
		}

		// read net-info.json created for pod
		podDir := filepath.Join(ctx.DataDir(), "pods", "run", string(podUUID))
		podDirfd, err := syscall.Open(podDir, syscall.O_RDONLY|syscall.O_DIRECTORY, 0)
		if err != nil {
			t.Fatalf("Can't open pod directory for reading! %v", err)
		}

		info, err := netinfo.LoadAt(podDirfd)
		if err != nil {
			t.Fatalf("Can't open net-info.json for reading: %v", err)
		}

		if len(info) != 2 {
			t.Fatalf("Incorrect number of networks: %v", len(info))
		}

		found := false
		for _, net := range info {
			if net.NetName == ntFlannel.Name {
				found = true
				break
			}
		}

		if !found {
			t.Fatalf("Network '%s' not found!\nnetInfo[0]: %v\nnetInfo[1]: %v", ntFlannel.Name, info[0], info[1])
		}
	})
}

/*
 * NewNetDefaultGWTest checks if default gateway is correct if only configured network is one provided by flannel.
 */
func NewNetDefaultGWTest() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
		ctx := testutils.NewRktRunCtx()
		defer ctx.Cleanup()

		_, ntFlannel, err := mockFlannelNetwork(t, ctx)
		if err != nil {
			t.Errorf("Can't mock flannel network: %v", err)
		}

		defer os.Remove(ntFlannel.SubnetFile)

		testImageArgs := []string{"--exec=/inspect --print-defaultgwv4"}
		testImage := patchTestACI("rkt-inspect-networking.aci", testImageArgs...)
		defer os.Remove(testImage)

		cmd := fmt.Sprintf("%s --debug --insecure-options=image run --net=%s --mds-register=false %s", ctx.Cmd(), ntFlannel.Name, testImage)
		child := spawnOrFail(t, cmd)
		defer waitOrFail(t, child, 0)

		expectedRegex := `DefaultGWv4: (\d+\.\d+\.\d+\.\d+)`
		if _, out, err := expectRegexTimeoutWithOutput(child, expectedRegex, time.Minute); err != nil {
			t.Fatalf("No default gateway!\nError: %v\nOutput: %v", err, out)
		}
	})
}
