/*
Copyright 2014 The Kubernetes Authors.

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

package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/tools/remotecommand"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubecontainertesting "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/server/portforward"
	remotecommandserver "k8s.io/kubernetes/pkg/kubelet/server/remotecommand"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	testUID = "9b01b80f-8fb4-11e4-95ab-4200af06647"
)

type fakeKubelet struct {
	podByNameFunc                      func(namespace, name string) (*v1.Pod, bool)
	containerInfoFunc                  func(podFullName string, uid types.UID, containerName string, req *cadvisorapi.ContainerInfoRequest) (*cadvisorapi.ContainerInfo, error)
	rawInfoFunc                        func(query *cadvisorapi.ContainerInfoRequest) (map[string]*cadvisorapi.ContainerInfo, error)
	machineInfoFunc                    func() (*cadvisorapi.MachineInfo, error)
	podsFunc                           func() []*v1.Pod
	runningPodsFunc                    func() ([]*v1.Pod, error)
	logFunc                            func(w http.ResponseWriter, req *http.Request)
	runFunc                            func(podFullName string, uid types.UID, containerName string, cmd []string) ([]byte, error)
	execFunc                           func(pod string, uid types.UID, container string, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool) error
	attachFunc                         func(pod string, uid types.UID, container string, in io.Reader, out, err io.WriteCloser, tty bool) error
	portForwardFunc                    func(name string, uid types.UID, port int32, stream io.ReadWriteCloser) error
	containerLogsFunc                  func(podFullName, containerName string, logOptions *v1.PodLogOptions, stdout, stderr io.Writer) error
	streamingConnectionIdleTimeoutFunc func() time.Duration
	hostnameFunc                       func() string
	resyncInterval                     time.Duration
	loopEntryTime                      time.Time
	plegHealth                         bool
	redirectURL                        *url.URL
}

func (fk *fakeKubelet) ResyncInterval() time.Duration {
	return fk.resyncInterval
}

func (fk *fakeKubelet) LatestLoopEntryTime() time.Time {
	return fk.loopEntryTime
}

func (fk *fakeKubelet) GetPodByName(namespace, name string) (*v1.Pod, bool) {
	return fk.podByNameFunc(namespace, name)
}

func (fk *fakeKubelet) GetContainerInfo(podFullName string, uid types.UID, containerName string, req *cadvisorapi.ContainerInfoRequest) (*cadvisorapi.ContainerInfo, error) {
	return fk.containerInfoFunc(podFullName, uid, containerName, req)
}

func (fk *fakeKubelet) GetRawContainerInfo(containerName string, req *cadvisorapi.ContainerInfoRequest, subcontainers bool) (map[string]*cadvisorapi.ContainerInfo, error) {
	return fk.rawInfoFunc(req)
}

func (fk *fakeKubelet) GetCachedMachineInfo() (*cadvisorapi.MachineInfo, error) {
	return fk.machineInfoFunc()
}

func (_ *fakeKubelet) GetVersionInfo() (*cadvisorapi.VersionInfo, error) {
	return &cadvisorapi.VersionInfo{}, nil
}

func (fk *fakeKubelet) GetPods() []*v1.Pod {
	return fk.podsFunc()
}

func (fk *fakeKubelet) GetRunningPods() ([]*v1.Pod, error) {
	return fk.runningPodsFunc()
}

func (fk *fakeKubelet) ServeLogs(w http.ResponseWriter, req *http.Request) {
	fk.logFunc(w, req)
}

func (fk *fakeKubelet) GetKubeletContainerLogs(podFullName, containerName string, logOptions *v1.PodLogOptions, stdout, stderr io.Writer) error {
	return fk.containerLogsFunc(podFullName, containerName, logOptions, stdout, stderr)
}

func (fk *fakeKubelet) GetHostname() string {
	return fk.hostnameFunc()
}

func (fk *fakeKubelet) RunInContainer(podFullName string, uid types.UID, containerName string, cmd []string) ([]byte, error) {
	return fk.runFunc(podFullName, uid, containerName, cmd)
}

func (fk *fakeKubelet) ExecInContainer(name string, uid types.UID, container string, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize, timeout time.Duration) error {
	return fk.execFunc(name, uid, container, cmd, in, out, err, tty)
}

func (fk *fakeKubelet) AttachContainer(name string, uid types.UID, container string, in io.Reader, out, err io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize) error {
	return fk.attachFunc(name, uid, container, in, out, err, tty)
}

func (fk *fakeKubelet) PortForward(name string, uid types.UID, port int32, stream io.ReadWriteCloser) error {
	return fk.portForwardFunc(name, uid, port, stream)
}

func (fk *fakeKubelet) GetExec(podFullName string, podUID types.UID, containerName string, cmd []string, streamOpts remotecommandserver.Options) (*url.URL, error) {
	return fk.redirectURL, nil
}

func (fk *fakeKubelet) GetAttach(podFullName string, podUID types.UID, containerName string, streamOpts remotecommandserver.Options) (*url.URL, error) {
	return fk.redirectURL, nil
}

func (fk *fakeKubelet) GetPortForward(podName, podNamespace string, podUID types.UID, portForwardOpts portforward.V4Options) (*url.URL, error) {
	return fk.redirectURL, nil
}

func (fk *fakeKubelet) StreamingConnectionIdleTimeout() time.Duration {
	return fk.streamingConnectionIdleTimeoutFunc()
}

// Unused functions
func (_ *fakeKubelet) GetContainerInfoV2(_ string, _ cadvisorapiv2.RequestOptions) (map[string]cadvisorapiv2.ContainerInfo, error) {
	return nil, nil
}

func (_ *fakeKubelet) ImagesFsInfo() (cadvisorapiv2.FsInfo, error) {
	return cadvisorapiv2.FsInfo{}, fmt.Errorf("Unsupported Operation ImagesFsInfo")
}

func (_ *fakeKubelet) RootFsInfo() (cadvisorapiv2.FsInfo, error) {
	return cadvisorapiv2.FsInfo{}, fmt.Errorf("Unsupport Operation RootFsInfo")
}

func (_ *fakeKubelet) GetNode() (*v1.Node, error)   { return nil, nil }
func (_ *fakeKubelet) GetNodeConfig() cm.NodeConfig { return cm.NodeConfig{} }

func (fk *fakeKubelet) ListVolumesForPod(podUID types.UID) (map[string]volume.Volume, bool) {
	return map[string]volume.Volume{}, true
}

type fakeAuth struct {
	authenticateFunc func(*http.Request) (user.Info, bool, error)
	attributesFunc   func(user.Info, *http.Request) authorizer.Attributes
	authorizeFunc    func(authorizer.Attributes) (authorized bool, reason string, err error)
}

func (f *fakeAuth) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	return f.authenticateFunc(req)
}
func (f *fakeAuth) GetRequestAttributes(u user.Info, req *http.Request) authorizer.Attributes {
	return f.attributesFunc(u, req)
}
func (f *fakeAuth) Authorize(a authorizer.Attributes) (authorized bool, reason string, err error) {
	return f.authorizeFunc(a)
}

type serverTestFramework struct {
	serverUnderTest *Server
	fakeKubelet     *fakeKubelet
	fakeAuth        *fakeAuth
	testHTTPServer  *httptest.Server
	criHandler      *utiltesting.FakeHandler
}

func newServerTest() *serverTestFramework {
	fw := &serverTestFramework{}
	fw.fakeKubelet = &fakeKubelet{
		hostnameFunc: func() string {
			return "127.0.0.1"
		},
		podByNameFunc: func(namespace, name string) (*v1.Pod, bool) {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: namespace,
					Name:      name,
					UID:       testUID,
				},
			}, true
		},
		plegHealth: true,
	}
	fw.fakeAuth = &fakeAuth{
		authenticateFunc: func(req *http.Request) (user.Info, bool, error) {
			return &user.DefaultInfo{Name: "test"}, true, nil
		},
		attributesFunc: func(u user.Info, req *http.Request) authorizer.Attributes {
			return &authorizer.AttributesRecord{User: u}
		},
		authorizeFunc: func(a authorizer.Attributes) (authorized bool, reason string, err error) {
			return true, "", nil
		},
	}
	fw.criHandler = &utiltesting.FakeHandler{
		StatusCode: http.StatusOK,
	}
	server := NewServer(
		fw.fakeKubelet,
		stats.NewResourceAnalyzer(fw.fakeKubelet, time.Minute, &kubecontainertesting.FakeRuntime{}),
		fw.fakeAuth,
		true,
		false,
		&kubecontainertesting.Mock{},
		fw.criHandler)
	fw.serverUnderTest = &server
	fw.testHTTPServer = httptest.NewServer(fw.serverUnderTest)
	return fw
}

// encodeJSON returns obj marshalled as a JSON string, panicing on any errors
func encodeJSON(obj interface{}) string {
	data, err := json.Marshal(obj)
	if err != nil {
		panic(err)
	}
	return string(data)
}

func readResp(resp *http.Response) (string, error) {
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	return string(body), err
}

// A helper function to return the correct pod name.
func getPodName(name, namespace string) string {
	if namespace == "" {
		namespace = metav1.NamespaceDefault
	}
	return name + "_" + namespace
}

func TestContainerInfo(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	expectedInfo := &cadvisorapi.ContainerInfo{}
	podID := "somepod"
	expectedPodID := getPodName(podID, "")
	expectedContainerName := "goodcontainer"
	fw.fakeKubelet.containerInfoFunc = func(podID string, uid types.UID, containerName string, req *cadvisorapi.ContainerInfoRequest) (*cadvisorapi.ContainerInfo, error) {
		if podID != expectedPodID || containerName != expectedContainerName {
			return nil, fmt.Errorf("bad podID or containerName: podID=%v; containerName=%v", podID, containerName)
		}
		return expectedInfo, nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + fmt.Sprintf("/stats/%v/%v", podID, expectedContainerName))
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	var receivedInfo cadvisorapi.ContainerInfo
	err = json.NewDecoder(resp.Body).Decode(&receivedInfo)
	if err != nil {
		t.Fatalf("received invalid json data: %v", err)
	}
	if !receivedInfo.Eq(expectedInfo) {
		t.Errorf("received wrong data: %#v", receivedInfo)
	}
}

func TestContainerInfoWithUidNamespace(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	expectedInfo := &cadvisorapi.ContainerInfo{}
	podID := "somepod"
	expectedNamespace := "custom"
	expectedPodID := getPodName(podID, expectedNamespace)
	expectedContainerName := "goodcontainer"
	fw.fakeKubelet.containerInfoFunc = func(podID string, uid types.UID, containerName string, req *cadvisorapi.ContainerInfoRequest) (*cadvisorapi.ContainerInfo, error) {
		if podID != expectedPodID || string(uid) != testUID || containerName != expectedContainerName {
			return nil, fmt.Errorf("bad podID or uid or containerName: podID=%v; uid=%v; containerName=%v", podID, uid, containerName)
		}
		return expectedInfo, nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + fmt.Sprintf("/stats/%v/%v/%v/%v", expectedNamespace, podID, testUID, expectedContainerName))
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	var receivedInfo cadvisorapi.ContainerInfo
	err = json.NewDecoder(resp.Body).Decode(&receivedInfo)
	if err != nil {
		t.Fatalf("received invalid json data: %v", err)
	}
	if !receivedInfo.Eq(expectedInfo) {
		t.Errorf("received wrong data: %#v", receivedInfo)
	}
}

func TestContainerNotFound(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	podID := "somepod"
	expectedNamespace := "custom"
	expectedContainerName := "slowstartcontainer"
	fw.fakeKubelet.containerInfoFunc = func(podID string, uid types.UID, containerName string, req *cadvisorapi.ContainerInfoRequest) (*cadvisorapi.ContainerInfo, error) {
		return nil, kubecontainer.ErrContainerNotFound
	}
	resp, err := http.Get(fw.testHTTPServer.URL + fmt.Sprintf("/stats/%v/%v/%v/%v", expectedNamespace, podID, testUID, expectedContainerName))
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("Received status %d expecting %d", resp.StatusCode, http.StatusNotFound)
	}
	defer resp.Body.Close()
}

func TestRootInfo(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	expectedInfo := &cadvisorapi.ContainerInfo{
		ContainerReference: cadvisorapi.ContainerReference{
			Name: "/",
		},
	}
	fw.fakeKubelet.rawInfoFunc = func(req *cadvisorapi.ContainerInfoRequest) (map[string]*cadvisorapi.ContainerInfo, error) {
		return map[string]*cadvisorapi.ContainerInfo{
			expectedInfo.Name: expectedInfo,
		}, nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + "/stats")
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	var receivedInfo cadvisorapi.ContainerInfo
	err = json.NewDecoder(resp.Body).Decode(&receivedInfo)
	if err != nil {
		t.Fatalf("received invalid json data: %v", err)
	}
	if !receivedInfo.Eq(expectedInfo) {
		t.Errorf("received wrong data: %#v, expected %#v", receivedInfo, expectedInfo)
	}
}

func TestSubcontainerContainerInfo(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	const kubeletContainer = "/kubelet"
	const kubeletSubContainer = "/kubelet/sub"
	expectedInfo := map[string]*cadvisorapi.ContainerInfo{
		kubeletContainer: {
			ContainerReference: cadvisorapi.ContainerReference{
				Name: kubeletContainer,
			},
		},
		kubeletSubContainer: {
			ContainerReference: cadvisorapi.ContainerReference{
				Name: kubeletSubContainer,
			},
		},
	}
	fw.fakeKubelet.rawInfoFunc = func(req *cadvisorapi.ContainerInfoRequest) (map[string]*cadvisorapi.ContainerInfo, error) {
		return expectedInfo, nil
	}

	request := fmt.Sprintf("{\"containerName\":%q, \"subcontainers\": true}", kubeletContainer)
	resp, err := http.Post(fw.testHTTPServer.URL+"/stats/container", "application/json", bytes.NewBuffer([]byte(request)))
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	var receivedInfo map[string]*cadvisorapi.ContainerInfo
	err = json.NewDecoder(resp.Body).Decode(&receivedInfo)
	if err != nil {
		t.Fatalf("Received invalid json data: %v", err)
	}
	if len(receivedInfo) != len(expectedInfo) {
		t.Errorf("Received wrong data: %#v, expected %#v", receivedInfo, expectedInfo)
	}

	for _, containerName := range []string{kubeletContainer, kubeletSubContainer} {
		if _, ok := receivedInfo[containerName]; !ok {
			t.Errorf("Expected container %q to be present in result: %#v", containerName, receivedInfo)
		}
		if !receivedInfo[containerName].Eq(expectedInfo[containerName]) {
			t.Errorf("Invalid result for %q: Expected %#v, received %#v", containerName, expectedInfo[containerName], receivedInfo[containerName])
		}
	}
}

func TestMachineInfo(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	expectedInfo := &cadvisorapi.MachineInfo{
		NumCores:       4,
		MemoryCapacity: 1024,
	}
	fw.fakeKubelet.machineInfoFunc = func() (*cadvisorapi.MachineInfo, error) {
		return expectedInfo, nil
	}

	resp, err := http.Get(fw.testHTTPServer.URL + "/spec")
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	var receivedInfo cadvisorapi.MachineInfo
	err = json.NewDecoder(resp.Body).Decode(&receivedInfo)
	if err != nil {
		t.Fatalf("received invalid json data: %v", err)
	}
	if !reflect.DeepEqual(&receivedInfo, expectedInfo) {
		t.Errorf("received wrong data: %#v", receivedInfo)
	}
}

func TestServeLogs(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()

	content := string(`<pre><a href="kubelet.log">kubelet.log</a><a href="google.log">google.log</a></pre>`)

	fw.fakeKubelet.logFunc = func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Header().Add("Content-Type", "text/html")
		w.Write([]byte(content))
	}

	resp, err := http.Get(fw.testHTTPServer.URL + "/logs/")
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()

	body, err := httputil.DumpResponse(resp, true)
	if err != nil {
		// copying the response body did not work
		t.Errorf("Cannot copy resp: %#v", err)
	}
	result := string(body)
	if !strings.Contains(result, "kubelet.log") || !strings.Contains(result, "google.log") {
		t.Errorf("Received wrong data: %s", result)
	}
}

func TestServeRunInContainer(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	output := "foo bar"
	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)
	expectedContainerName := "baz"
	expectedCommand := "ls -a"
	fw.fakeKubelet.runFunc = func(podFullName string, uid types.UID, containerName string, cmd []string) ([]byte, error) {
		if podFullName != expectedPodName {
			t.Errorf("expected %s, got %s", expectedPodName, podFullName)
		}
		if containerName != expectedContainerName {
			t.Errorf("expected %s, got %s", expectedContainerName, containerName)
		}
		if strings.Join(cmd, " ") != expectedCommand {
			t.Errorf("expected: %s, got %v", expectedCommand, cmd)
		}

		return []byte(output), nil
	}

	resp, err := http.Post(fw.testHTTPServer.URL+"/run/"+podNamespace+"/"+podName+"/"+expectedContainerName+"?cmd=ls%20-a", "", nil)

	if err != nil {
		t.Fatalf("Got error POSTing: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		// copying the response body did not work
		t.Errorf("Cannot copy resp: %#v", err)
	}
	result := string(body)
	if result != output {
		t.Errorf("expected %s, got %s", output, result)
	}
}

func TestServeRunInContainerWithUID(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	output := "foo bar"
	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)
	expectedContainerName := "baz"
	expectedCommand := "ls -a"
	fw.fakeKubelet.runFunc = func(podFullName string, uid types.UID, containerName string, cmd []string) ([]byte, error) {
		if podFullName != expectedPodName {
			t.Errorf("expected %s, got %s", expectedPodName, podFullName)
		}
		if string(uid) != testUID {
			t.Errorf("expected %s, got %s", testUID, uid)
		}
		if containerName != expectedContainerName {
			t.Errorf("expected %s, got %s", expectedContainerName, containerName)
		}
		if strings.Join(cmd, " ") != expectedCommand {
			t.Errorf("expected: %s, got %v", expectedCommand, cmd)
		}

		return []byte(output), nil
	}

	resp, err := http.Post(fw.testHTTPServer.URL+"/run/"+podNamespace+"/"+podName+"/"+testUID+"/"+expectedContainerName+"?cmd=ls%20-a", "", nil)

	if err != nil {
		t.Fatalf("Got error POSTing: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		// copying the response body did not work
		t.Errorf("Cannot copy resp: %#v", err)
	}
	result := string(body)
	if result != output {
		t.Errorf("expected %s, got %s", output, result)
	}
}

func TestHealthCheck(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	fw.fakeKubelet.hostnameFunc = func() string {
		return "127.0.0.1"
	}

	// Test with correct hostname, Docker version
	assertHealthIsOk(t, fw.testHTTPServer.URL+"/healthz")

	// Test with incorrect hostname
	fw.fakeKubelet.hostnameFunc = func() string {
		return "fake"
	}
	assertHealthIsOk(t, fw.testHTTPServer.URL+"/healthz")
}

func assertHealthFails(t *testing.T, httpURL string, expectedErrorCode int) {
	resp, err := http.Get(httpURL)
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != expectedErrorCode {
		t.Errorf("expected status code %d, got %d", expectedErrorCode, resp.StatusCode)
	}
}

type authTestCase struct {
	Method string
	Path   string
}

func TestAuthFilters(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()

	testcases := []authTestCase{}

	// This is a sanity check that the Handle->HandleWithFilter() delegation is working
	// Ideally, these would move to registered web services and this list would get shorter
	expectedPaths := []string{"/healthz", "/metrics", "/metrics/cadvisor"}
	paths := sets.NewString(fw.serverUnderTest.restfulCont.RegisteredHandlePaths()...)
	for _, expectedPath := range expectedPaths {
		if !paths.Has(expectedPath) {
			t.Errorf("Expected registered handle path %s was missing", expectedPath)
		}
	}

	// Test all the non-web-service handlers
	for _, path := range fw.serverUnderTest.restfulCont.RegisteredHandlePaths() {
		testcases = append(testcases, authTestCase{"GET", path})
		testcases = append(testcases, authTestCase{"POST", path})
		// Test subpaths for directory handlers
		if strings.HasSuffix(path, "/") {
			testcases = append(testcases, authTestCase{"GET", path + "foo"})
			testcases = append(testcases, authTestCase{"POST", path + "foo"})
		}
	}

	// Test all the generated web-service paths
	for _, ws := range fw.serverUnderTest.restfulCont.RegisteredWebServices() {
		for _, r := range ws.Routes() {
			testcases = append(testcases, authTestCase{r.Method, r.Path})
		}
	}

	methodToAPIVerb := map[string]string{"GET": "get", "POST": "create", "PUT": "update"}
	pathToSubresource := func(path string) string {
		switch {
		// Cases for subpaths we expect specific subresources for
		case isSubpath(path, statsPath):
			return "stats"
		case isSubpath(path, specPath):
			return "spec"
		case isSubpath(path, logsPath):
			return "log"
		case isSubpath(path, metricsPath):
			return "metrics"

		// Cases for subpaths we expect to map to the "proxy" subresource
		case isSubpath(path, "/attach"),
			isSubpath(path, "/configz"),
			isSubpath(path, "/containerLogs"),
			isSubpath(path, "/debug"),
			isSubpath(path, "/exec"),
			isSubpath(path, "/healthz"),
			isSubpath(path, "/pods"),
			isSubpath(path, "/portForward"),
			isSubpath(path, "/run"),
			isSubpath(path, "/runningpods"),
			isSubpath(path, "/cri"):
			return "proxy"

		default:
			panic(fmt.Errorf(`unexpected kubelet API path %s.
The kubelet API has likely registered a handler for a new path.
If the new path has a use case for partitioned authorization when requested from the kubelet API,
add a specific subresource for it in auth.go#GetRequestAttributes() and in TestAuthFilters().
Otherwise, add it to the expected list of paths that map to the "proxy" subresource in TestAuthFilters().`, path))
		}
	}
	attributesGetter := NewNodeAuthorizerAttributesGetter(types.NodeName("test"))

	for _, tc := range testcases {
		var (
			expectedUser       = &user.DefaultInfo{Name: "test"}
			expectedAttributes = authorizer.AttributesRecord{
				User:            expectedUser,
				APIGroup:        "",
				APIVersion:      "v1",
				Verb:            methodToAPIVerb[tc.Method],
				Resource:        "nodes",
				Name:            "test",
				Subresource:     pathToSubresource(tc.Path),
				ResourceRequest: true,
				Path:            tc.Path,
			}

			calledAuthenticate = false
			calledAuthorize    = false
			calledAttributes   = false
		)

		fw.fakeAuth.authenticateFunc = func(req *http.Request) (user.Info, bool, error) {
			calledAuthenticate = true
			return expectedUser, true, nil
		}
		fw.fakeAuth.attributesFunc = func(u user.Info, req *http.Request) authorizer.Attributes {
			calledAttributes = true
			if u != expectedUser {
				t.Fatalf("%s: expected user %v, got %v", tc.Path, expectedUser, u)
			}
			return attributesGetter.GetRequestAttributes(u, req)
		}
		fw.fakeAuth.authorizeFunc = func(a authorizer.Attributes) (authorized bool, reason string, err error) {
			calledAuthorize = true
			if a != expectedAttributes {
				t.Fatalf("%s: expected attributes\n\t%#v\ngot\n\t%#v", tc.Path, expectedAttributes, a)
			}
			return false, "", nil
		}

		req, err := http.NewRequest(tc.Method, fw.testHTTPServer.URL+tc.Path, nil)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", tc.Path, err)
			continue
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", tc.Path, err)
			continue
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusForbidden {
			t.Errorf("%s: unexpected status code %d", tc.Path, resp.StatusCode)
			continue
		}

		if !calledAuthenticate {
			t.Errorf("%s: Authenticate was not called", tc.Path)
			continue
		}
		if !calledAttributes {
			t.Errorf("%s: Attributes were not called", tc.Path)
			continue
		}
		if !calledAuthorize {
			t.Errorf("%s: Authorize was not called", tc.Path)
			continue
		}
	}
}

func TestAuthenticationError(t *testing.T) {
	var (
		expectedUser       = &user.DefaultInfo{Name: "test"}
		expectedAttributes = &authorizer.AttributesRecord{User: expectedUser}

		calledAuthenticate = false
		calledAuthorize    = false
		calledAttributes   = false
	)

	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	fw.fakeAuth.authenticateFunc = func(req *http.Request) (user.Info, bool, error) {
		calledAuthenticate = true
		return expectedUser, true, nil
	}
	fw.fakeAuth.attributesFunc = func(u user.Info, req *http.Request) authorizer.Attributes {
		calledAttributes = true
		return expectedAttributes
	}
	fw.fakeAuth.authorizeFunc = func(a authorizer.Attributes) (authorized bool, reason string, err error) {
		calledAuthorize = true
		return false, "", errors.New("Failed")
	}

	assertHealthFails(t, fw.testHTTPServer.URL+"/healthz", http.StatusInternalServerError)

	if !calledAuthenticate {
		t.Fatalf("Authenticate was not called")
	}
	if !calledAttributes {
		t.Fatalf("Attributes was not called")
	}
	if !calledAuthorize {
		t.Fatalf("Authorize was not called")
	}
}

func TestAuthenticationFailure(t *testing.T) {
	var (
		expectedUser       = &user.DefaultInfo{Name: "test"}
		expectedAttributes = &authorizer.AttributesRecord{User: expectedUser}

		calledAuthenticate = false
		calledAuthorize    = false
		calledAttributes   = false
	)

	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	fw.fakeAuth.authenticateFunc = func(req *http.Request) (user.Info, bool, error) {
		calledAuthenticate = true
		return nil, false, nil
	}
	fw.fakeAuth.attributesFunc = func(u user.Info, req *http.Request) authorizer.Attributes {
		calledAttributes = true
		return expectedAttributes
	}
	fw.fakeAuth.authorizeFunc = func(a authorizer.Attributes) (authorized bool, reason string, err error) {
		calledAuthorize = true
		return false, "", nil
	}

	assertHealthFails(t, fw.testHTTPServer.URL+"/healthz", http.StatusUnauthorized)

	if !calledAuthenticate {
		t.Fatalf("Authenticate was not called")
	}
	if calledAttributes {
		t.Fatalf("Attributes was called unexpectedly")
	}
	if calledAuthorize {
		t.Fatalf("Authorize was called unexpectedly")
	}
}

func TestAuthorizationSuccess(t *testing.T) {
	var (
		expectedUser       = &user.DefaultInfo{Name: "test"}
		expectedAttributes = &authorizer.AttributesRecord{User: expectedUser}

		calledAuthenticate = false
		calledAuthorize    = false
		calledAttributes   = false
	)

	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	fw.fakeAuth.authenticateFunc = func(req *http.Request) (user.Info, bool, error) {
		calledAuthenticate = true
		return expectedUser, true, nil
	}
	fw.fakeAuth.attributesFunc = func(u user.Info, req *http.Request) authorizer.Attributes {
		calledAttributes = true
		return expectedAttributes
	}
	fw.fakeAuth.authorizeFunc = func(a authorizer.Attributes) (authorized bool, reason string, err error) {
		calledAuthorize = true
		return true, "", nil
	}

	assertHealthIsOk(t, fw.testHTTPServer.URL+"/healthz")

	if !calledAuthenticate {
		t.Fatalf("Authenticate was not called")
	}
	if !calledAttributes {
		t.Fatalf("Attributes were not called")
	}
	if !calledAuthorize {
		t.Fatalf("Authorize was not called")
	}
}

func TestSyncLoopCheck(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	fw.fakeKubelet.hostnameFunc = func() string {
		return "127.0.0.1"
	}

	fw.fakeKubelet.resyncInterval = time.Minute
	fw.fakeKubelet.loopEntryTime = time.Now()

	// Test with correct hostname, Docker version
	assertHealthIsOk(t, fw.testHTTPServer.URL+"/healthz")

	fw.fakeKubelet.loopEntryTime = time.Now().Add(time.Minute * -10)
	assertHealthFails(t, fw.testHTTPServer.URL+"/healthz", http.StatusInternalServerError)
}

// returns http response status code from the HTTP GET
func assertHealthIsOk(t *testing.T, httpURL string) {
	resp, err := http.Get(httpURL)
	if err != nil {
		t.Fatalf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected status code %d, got %d", http.StatusOK, resp.StatusCode)
	}
	body, readErr := ioutil.ReadAll(resp.Body)
	if readErr != nil {
		// copying the response body did not work
		t.Fatalf("Cannot copy resp: %#v", readErr)
	}
	result := string(body)
	if !strings.Contains(result, "ok") {
		t.Errorf("expected body contains ok, got %s", result)
	}
}

func setPodByNameFunc(fw *serverTestFramework, namespace, pod, container string) {
	fw.fakeKubelet.podByNameFunc = func(namespace, name string) (*v1.Pod, bool) {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: namespace,
				Name:      pod,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: container,
					},
				},
			},
		}, true
	}
}

func setGetContainerLogsFunc(fw *serverTestFramework, t *testing.T, expectedPodName, expectedContainerName string, expectedLogOptions *v1.PodLogOptions, output string) {
	fw.fakeKubelet.containerLogsFunc = func(podFullName, containerName string, logOptions *v1.PodLogOptions, stdout, stderr io.Writer) error {
		if podFullName != expectedPodName {
			t.Errorf("expected %s, got %s", expectedPodName, podFullName)
		}
		if containerName != expectedContainerName {
			t.Errorf("expected %s, got %s", expectedContainerName, containerName)
		}
		if !reflect.DeepEqual(expectedLogOptions, logOptions) {
			t.Errorf("expected %#v, got %#v", expectedLogOptions, logOptions)
		}

		io.WriteString(stdout, output)
		return nil
	}
}

// TODO: I really want to be a table driven test
func TestContainerLogs(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	output := "foo bar"
	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)
	expectedContainerName := "baz"
	setPodByNameFunc(fw, podNamespace, podName, expectedContainerName)
	setGetContainerLogsFunc(fw, t, expectedPodName, expectedContainerName, &v1.PodLogOptions{}, output)
	resp, err := http.Get(fw.testHTTPServer.URL + "/containerLogs/" + podNamespace + "/" + podName + "/" + expectedContainerName)
	if err != nil {
		t.Errorf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Errorf("Error reading container logs: %v", err)
	}
	result := string(body)
	if result != output {
		t.Errorf("Expected: '%v', got: '%v'", output, result)
	}
}

func TestContainerLogsWithLimitBytes(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	output := "foo bar"
	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)
	expectedContainerName := "baz"
	bytes := int64(3)
	setPodByNameFunc(fw, podNamespace, podName, expectedContainerName)
	setGetContainerLogsFunc(fw, t, expectedPodName, expectedContainerName, &v1.PodLogOptions{LimitBytes: &bytes}, output)
	resp, err := http.Get(fw.testHTTPServer.URL + "/containerLogs/" + podNamespace + "/" + podName + "/" + expectedContainerName + "?limitBytes=3")
	if err != nil {
		t.Errorf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Errorf("Error reading container logs: %v", err)
	}
	result := string(body)
	if result != output[:bytes] {
		t.Errorf("Expected: '%v', got: '%v'", output[:bytes], result)
	}
}

func TestContainerLogsWithTail(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	output := "foo bar"
	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)
	expectedContainerName := "baz"
	expectedTail := int64(5)
	setPodByNameFunc(fw, podNamespace, podName, expectedContainerName)
	setGetContainerLogsFunc(fw, t, expectedPodName, expectedContainerName, &v1.PodLogOptions{TailLines: &expectedTail}, output)
	resp, err := http.Get(fw.testHTTPServer.URL + "/containerLogs/" + podNamespace + "/" + podName + "/" + expectedContainerName + "?tailLines=5")
	if err != nil {
		t.Errorf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Errorf("Error reading container logs: %v", err)
	}
	result := string(body)
	if result != output {
		t.Errorf("Expected: '%v', got: '%v'", output, result)
	}
}

func TestContainerLogsWithLegacyTail(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	output := "foo bar"
	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)
	expectedContainerName := "baz"
	expectedTail := int64(5)
	setPodByNameFunc(fw, podNamespace, podName, expectedContainerName)
	setGetContainerLogsFunc(fw, t, expectedPodName, expectedContainerName, &v1.PodLogOptions{TailLines: &expectedTail}, output)
	resp, err := http.Get(fw.testHTTPServer.URL + "/containerLogs/" + podNamespace + "/" + podName + "/" + expectedContainerName + "?tail=5")
	if err != nil {
		t.Errorf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Errorf("Error reading container logs: %v", err)
	}
	result := string(body)
	if result != output {
		t.Errorf("Expected: '%v', got: '%v'", output, result)
	}
}

func TestContainerLogsWithTailAll(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	output := "foo bar"
	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)
	expectedContainerName := "baz"
	setPodByNameFunc(fw, podNamespace, podName, expectedContainerName)
	setGetContainerLogsFunc(fw, t, expectedPodName, expectedContainerName, &v1.PodLogOptions{}, output)
	resp, err := http.Get(fw.testHTTPServer.URL + "/containerLogs/" + podNamespace + "/" + podName + "/" + expectedContainerName + "?tail=all")
	if err != nil {
		t.Errorf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Errorf("Error reading container logs: %v", err)
	}
	result := string(body)
	if result != output {
		t.Errorf("Expected: '%v', got: '%v'", output, result)
	}
}

func TestContainerLogsWithInvalidTail(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	output := "foo bar"
	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)
	expectedContainerName := "baz"
	setPodByNameFunc(fw, podNamespace, podName, expectedContainerName)
	setGetContainerLogsFunc(fw, t, expectedPodName, expectedContainerName, &v1.PodLogOptions{}, output)
	resp, err := http.Get(fw.testHTTPServer.URL + "/containerLogs/" + podNamespace + "/" + podName + "/" + expectedContainerName + "?tail=-1")
	if err != nil {
		t.Errorf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != apierrs.StatusUnprocessableEntity {
		t.Errorf("Unexpected non-error reading container logs: %#v", resp)
	}
}

func TestContainerLogsWithFollow(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()
	output := "foo bar"
	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)
	expectedContainerName := "baz"
	setPodByNameFunc(fw, podNamespace, podName, expectedContainerName)
	setGetContainerLogsFunc(fw, t, expectedPodName, expectedContainerName, &v1.PodLogOptions{Follow: true}, output)
	resp, err := http.Get(fw.testHTTPServer.URL + "/containerLogs/" + podNamespace + "/" + podName + "/" + expectedContainerName + "?follow=1")
	if err != nil {
		t.Errorf("Got error GETing: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Errorf("Error reading container logs: %v", err)
	}
	result := string(body)
	if result != output {
		t.Errorf("Expected: '%v', got: '%v'", output, result)
	}
}

func TestServeExecInContainerIdleTimeout(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()

	fw.fakeKubelet.streamingConnectionIdleTimeoutFunc = func() time.Duration {
		return 100 * time.Millisecond
	}

	podNamespace := "other"
	podName := "foo"
	expectedContainerName := "baz"

	url := fw.testHTTPServer.URL + "/exec/" + podNamespace + "/" + podName + "/" + expectedContainerName + "?c=ls&c=-a&" + api.ExecStdinParam + "=1"

	upgradeRoundTripper := spdy.NewSpdyRoundTripper(nil, true)
	c := &http.Client{Transport: upgradeRoundTripper}

	resp, err := c.Post(url, "", nil)
	if err != nil {
		t.Fatalf("Got error POSTing: %v", err)
	}
	defer resp.Body.Close()

	upgradeRoundTripper.Dialer = &net.Dialer{
		Deadline: time.Now().Add(60 * time.Second),
		Timeout:  60 * time.Second,
	}
	conn, err := upgradeRoundTripper.NewConnection(resp)
	if err != nil {
		t.Fatalf("Unexpected error creating streaming connection: %s", err)
	}
	if conn == nil {
		t.Fatal("Unexpected nil connection")
	}

	<-conn.CloseChan()
}

func testExecAttach(t *testing.T, verb string) {
	tests := []struct {
		stdin              bool
		stdout             bool
		stderr             bool
		tty                bool
		responseStatusCode int
		uid                bool
		responseLocation   string
	}{
		{responseStatusCode: http.StatusBadRequest},
		{stdin: true, responseStatusCode: http.StatusSwitchingProtocols},
		{stdout: true, responseStatusCode: http.StatusSwitchingProtocols},
		{stderr: true, responseStatusCode: http.StatusSwitchingProtocols},
		{stdout: true, stderr: true, responseStatusCode: http.StatusSwitchingProtocols},
		{stdout: true, stderr: true, tty: true, responseStatusCode: http.StatusSwitchingProtocols},
		{stdin: true, stdout: true, stderr: true, responseStatusCode: http.StatusSwitchingProtocols},
		{stdout: true, responseStatusCode: http.StatusFound, responseLocation: "http://localhost:12345/" + verb},
	}

	for i, test := range tests {
		fw := newServerTest()
		defer fw.testHTTPServer.Close()

		fw.fakeKubelet.streamingConnectionIdleTimeoutFunc = func() time.Duration {
			return 0
		}

		if test.responseLocation != "" {
			var err error
			fw.fakeKubelet.redirectURL, err = url.Parse(test.responseLocation)
			require.NoError(t, err)
		}

		podNamespace := "other"
		podName := "foo"
		expectedPodName := getPodName(podName, podNamespace)
		expectedContainerName := "baz"
		expectedCommand := "ls -a"
		expectedStdin := "stdin"
		expectedStdout := "stdout"
		expectedStderr := "stderr"
		done := make(chan struct{})
		clientStdoutReadDone := make(chan struct{})
		clientStderrReadDone := make(chan struct{})
		execInvoked := false
		attachInvoked := false

		testStreamFunc := func(podFullName string, uid types.UID, containerName string, cmd []string, in io.Reader, out, stderr io.WriteCloser, tty bool, done chan struct{}) error {
			defer close(done)

			if podFullName != expectedPodName {
				t.Fatalf("%d: podFullName: expected %s, got %s", i, expectedPodName, podFullName)
			}
			if test.uid && string(uid) != testUID {
				t.Fatalf("%d: uid: expected %v, got %v", i, testUID, uid)
			}
			if containerName != expectedContainerName {
				t.Fatalf("%d: containerName: expected %s, got %s", i, expectedContainerName, containerName)
			}

			if test.stdin {
				if in == nil {
					t.Fatalf("%d: stdin: expected non-nil", i)
				}
				b := make([]byte, 10)
				n, err := in.Read(b)
				if err != nil {
					t.Fatalf("%d: error reading from stdin: %v", i, err)
				}
				if e, a := expectedStdin, string(b[0:n]); e != a {
					t.Fatalf("%d: stdin: expected to read %v, got %v", i, e, a)
				}
			} else if in != nil {
				t.Fatalf("%d: stdin: expected nil: %#v", i, in)
			}

			if test.stdout {
				if out == nil {
					t.Fatalf("%d: stdout: expected non-nil", i)
				}
				_, err := out.Write([]byte(expectedStdout))
				if err != nil {
					t.Fatalf("%d:, error writing to stdout: %v", i, err)
				}
				out.Close()
				<-clientStdoutReadDone
			} else if out != nil {
				t.Fatalf("%d: stdout: expected nil: %#v", i, out)
			}

			if tty {
				if stderr != nil {
					t.Fatalf("%d: tty set but received non-nil stderr: %v", i, stderr)
				}
			} else if test.stderr {
				if stderr == nil {
					t.Fatalf("%d: stderr: expected non-nil", i)
				}
				_, err := stderr.Write([]byte(expectedStderr))
				if err != nil {
					t.Fatalf("%d:, error writing to stderr: %v", i, err)
				}
				stderr.Close()
				<-clientStderrReadDone
			} else if stderr != nil {
				t.Fatalf("%d: stderr: expected nil: %#v", i, stderr)
			}

			return nil
		}

		fw.fakeKubelet.execFunc = func(podFullName string, uid types.UID, containerName string, cmd []string, in io.Reader, out, stderr io.WriteCloser, tty bool) error {
			execInvoked = true
			if strings.Join(cmd, " ") != expectedCommand {
				t.Fatalf("%d: cmd: expected: %s, got %v", i, expectedCommand, cmd)
			}
			return testStreamFunc(podFullName, uid, containerName, cmd, in, out, stderr, tty, done)
		}

		fw.fakeKubelet.attachFunc = func(podFullName string, uid types.UID, containerName string, in io.Reader, out, stderr io.WriteCloser, tty bool) error {
			attachInvoked = true
			return testStreamFunc(podFullName, uid, containerName, nil, in, out, stderr, tty, done)
		}

		var url string
		if test.uid {
			url = fw.testHTTPServer.URL + "/" + verb + "/" + podNamespace + "/" + podName + "/" + testUID + "/" + expectedContainerName + "?ignore=1"
		} else {
			url = fw.testHTTPServer.URL + "/" + verb + "/" + podNamespace + "/" + podName + "/" + expectedContainerName + "?ignore=1"
		}
		if verb == "exec" {
			url += "&command=ls&command=-a"
		}
		if test.stdin {
			url += "&" + api.ExecStdinParam + "=1"
		}
		if test.stdout {
			url += "&" + api.ExecStdoutParam + "=1"
		}
		if test.stderr && !test.tty {
			url += "&" + api.ExecStderrParam + "=1"
		}
		if test.tty {
			url += "&" + api.ExecTTYParam + "=1"
		}

		var (
			resp                *http.Response
			err                 error
			upgradeRoundTripper httpstream.UpgradeRoundTripper
			c                   *http.Client
		)

		if test.responseStatusCode != http.StatusSwitchingProtocols {
			c = &http.Client{}
			// Don't follow redirects, since we want to inspect the redirect response.
			c.CheckRedirect = func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			}
		} else {
			upgradeRoundTripper = spdy.NewRoundTripper(nil, true)
			c = &http.Client{Transport: upgradeRoundTripper}
		}

		resp, err = c.Post(url, "", nil)
		if err != nil {
			t.Fatalf("%d: Got error POSTing: %v", i, err)
		}
		defer resp.Body.Close()

		_, err = ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("%d: Error reading response body: %v", i, err)
		}

		if e, a := test.responseStatusCode, resp.StatusCode; e != a {
			t.Fatalf("%d: response status: expected %v, got %v", i, e, a)
		}

		if e, a := test.responseLocation, resp.Header.Get("Location"); e != a {
			t.Errorf("%d: response location: expected %v, got %v", i, e, a)
		}

		if test.responseStatusCode != http.StatusSwitchingProtocols {
			continue
		}

		conn, err := upgradeRoundTripper.NewConnection(resp)
		if err != nil {
			t.Fatalf("Unexpected error creating streaming connection: %s", err)
		}
		if conn == nil {
			t.Fatalf("%d: unexpected nil conn", i)
		}
		defer conn.Close()

		h := http.Header{}
		h.Set(api.StreamType, api.StreamTypeError)
		if _, err := conn.CreateStream(h); err != nil {
			t.Fatalf("%d: error creating error stream: %v", i, err)
		}

		if test.stdin {
			h.Set(api.StreamType, api.StreamTypeStdin)
			stream, err := conn.CreateStream(h)
			if err != nil {
				t.Fatalf("%d: error creating stdin stream: %v", i, err)
			}
			_, err = stream.Write([]byte(expectedStdin))
			if err != nil {
				t.Fatalf("%d: error writing to stdin stream: %v", i, err)
			}
		}

		var stdoutStream httpstream.Stream
		if test.stdout {
			h.Set(api.StreamType, api.StreamTypeStdout)
			stdoutStream, err = conn.CreateStream(h)
			if err != nil {
				t.Fatalf("%d: error creating stdout stream: %v", i, err)
			}
		}

		var stderrStream httpstream.Stream
		if test.stderr && !test.tty {
			h.Set(api.StreamType, api.StreamTypeStderr)
			stderrStream, err = conn.CreateStream(h)
			if err != nil {
				t.Fatalf("%d: error creating stderr stream: %v", i, err)
			}
		}

		if test.stdout {
			output := make([]byte, 10)
			n, err := stdoutStream.Read(output)
			close(clientStdoutReadDone)
			if err != nil {
				t.Fatalf("%d: error reading from stdout stream: %v", i, err)
			}
			if e, a := expectedStdout, string(output[0:n]); e != a {
				t.Fatalf("%d: stdout: expected '%v', got '%v'", i, e, a)
			}
		}

		if test.stderr && !test.tty {
			output := make([]byte, 10)
			n, err := stderrStream.Read(output)
			close(clientStderrReadDone)
			if err != nil {
				t.Fatalf("%d: error reading from stderr stream: %v", i, err)
			}
			if e, a := expectedStderr, string(output[0:n]); e != a {
				t.Fatalf("%d: stderr: expected '%v', got '%v'", i, e, a)
			}
		}

		// wait for the server to finish before checking if the attach/exec funcs were invoked
		<-done

		if verb == "exec" {
			if !execInvoked {
				t.Errorf("%d: exec was not invoked", i)
			}
			if attachInvoked {
				t.Errorf("%d: attach should not have been invoked", i)
			}
		} else {
			if !attachInvoked {
				t.Errorf("%d: attach was not invoked", i)
			}
			if execInvoked {
				t.Errorf("%d: exec should not have been invoked", i)
			}
		}
	}
}

func TestServeExecInContainer(t *testing.T) {
	testExecAttach(t, "exec")
}

func TestServeAttachContainer(t *testing.T) {
	testExecAttach(t, "attach")
}

func TestServePortForwardIdleTimeout(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()

	fw.fakeKubelet.streamingConnectionIdleTimeoutFunc = func() time.Duration {
		return 100 * time.Millisecond
	}

	podNamespace := "other"
	podName := "foo"

	url := fw.testHTTPServer.URL + "/portForward/" + podNamespace + "/" + podName

	upgradeRoundTripper := spdy.NewRoundTripper(nil, true)
	c := &http.Client{Transport: upgradeRoundTripper}

	resp, err := c.Post(url, "", nil)
	if err != nil {
		t.Fatalf("Got error POSTing: %v", err)
	}
	defer resp.Body.Close()

	conn, err := upgradeRoundTripper.NewConnection(resp)
	if err != nil {
		t.Fatalf("Unexpected error creating streaming connection: %s", err)
	}
	if conn == nil {
		t.Fatal("Unexpected nil connection")
	}
	defer conn.Close()

	<-conn.CloseChan()
}

func TestServePortForward(t *testing.T) {
	tests := []struct {
		port             string
		uid              bool
		clientData       string
		containerData    string
		shouldError      bool
		responseLocation string
	}{
		{port: "", shouldError: true},
		{port: "abc", shouldError: true},
		{port: "-1", shouldError: true},
		{port: "65536", shouldError: true},
		{port: "0", shouldError: true},
		{port: "1", shouldError: false},
		{port: "8000", shouldError: false},
		{port: "8000", clientData: "client data", containerData: "container data", shouldError: false},
		{port: "65535", shouldError: false},
		{port: "65535", uid: true, shouldError: false},
		{port: "65535", responseLocation: "http://localhost:12345/portforward", shouldError: false},
	}

	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)

	for i, test := range tests {
		fw := newServerTest()
		defer fw.testHTTPServer.Close()

		fw.fakeKubelet.streamingConnectionIdleTimeoutFunc = func() time.Duration {
			return 0
		}

		if test.responseLocation != "" {
			var err error
			fw.fakeKubelet.redirectURL, err = url.Parse(test.responseLocation)
			require.NoError(t, err)
		}

		portForwardFuncDone := make(chan struct{})

		fw.fakeKubelet.portForwardFunc = func(name string, uid types.UID, port int32, stream io.ReadWriteCloser) error {
			defer close(portForwardFuncDone)

			if e, a := expectedPodName, name; e != a {
				t.Fatalf("%d: pod name: expected '%v', got '%v'", i, e, a)
			}

			if e, a := testUID, uid; test.uid && e != string(a) {
				t.Fatalf("%d: uid: expected '%v', got '%v'", i, e, a)
			}

			p, err := strconv.ParseInt(test.port, 10, 32)
			if err != nil {
				t.Fatalf("%d: error parsing port string '%s': %v", i, test.port, err)
			}
			if e, a := int32(p), port; e != a {
				t.Fatalf("%d: port: expected '%v', got '%v'", i, e, a)
			}

			if test.clientData != "" {
				fromClient := make([]byte, 32)
				n, err := stream.Read(fromClient)
				if err != nil {
					t.Fatalf("%d: error reading client data: %v", i, err)
				}
				if e, a := test.clientData, string(fromClient[0:n]); e != a {
					t.Fatalf("%d: client data: expected to receive '%v', got '%v'", i, e, a)
				}
			}

			if test.containerData != "" {
				_, err := stream.Write([]byte(test.containerData))
				if err != nil {
					t.Fatalf("%d: error writing container data: %v", i, err)
				}
			}

			return nil
		}

		var url string
		if test.uid {
			url = fmt.Sprintf("%s/portForward/%s/%s/%s", fw.testHTTPServer.URL, podNamespace, podName, testUID)
		} else {
			url = fmt.Sprintf("%s/portForward/%s/%s", fw.testHTTPServer.URL, podNamespace, podName)
		}

		var (
			upgradeRoundTripper httpstream.UpgradeRoundTripper
			c                   *http.Client
		)

		if len(test.responseLocation) > 0 {
			c = &http.Client{}
			// Don't follow redirects, since we want to inspect the redirect response.
			c.CheckRedirect = func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			}
		} else {
			upgradeRoundTripper = spdy.NewRoundTripper(nil, true)
			c = &http.Client{Transport: upgradeRoundTripper}
		}

		resp, err := c.Post(url, "", nil)
		if err != nil {
			t.Fatalf("%d: Got error POSTing: %v", i, err)
		}
		defer resp.Body.Close()

		if test.responseLocation != "" {
			assert.Equal(t, http.StatusFound, resp.StatusCode, "%d: status code", i)
			assert.Equal(t, test.responseLocation, resp.Header.Get("Location"), "%d: location", i)
			continue
		} else {
			assert.Equal(t, http.StatusSwitchingProtocols, resp.StatusCode, "%d: status code", i)
		}

		conn, err := upgradeRoundTripper.NewConnection(resp)
		if err != nil {
			t.Fatalf("Unexpected error creating streaming connection: %s", err)
		}
		if conn == nil {
			t.Fatalf("%d: Unexpected nil connection", i)
		}
		defer conn.Close()

		headers := http.Header{}
		headers.Set("streamType", "error")
		headers.Set("port", test.port)
		errorStream, err := conn.CreateStream(headers)
		_ = errorStream
		haveErr := err != nil
		if e, a := test.shouldError, haveErr; e != a {
			t.Fatalf("%d: create stream: expected err=%t, got %t: %v", i, e, a, err)
		}

		if test.shouldError {
			continue
		}

		headers.Set("streamType", "data")
		headers.Set("port", test.port)
		dataStream, err := conn.CreateStream(headers)
		haveErr = err != nil
		if e, a := test.shouldError, haveErr; e != a {
			t.Fatalf("%d: create stream: expected err=%t, got %t: %v", i, e, a, err)
		}

		if test.clientData != "" {
			_, err := dataStream.Write([]byte(test.clientData))
			if err != nil {
				t.Fatalf("%d: unexpected error writing client data: %v", i, err)
			}
		}

		if test.containerData != "" {
			fromContainer := make([]byte, 32)
			n, err := dataStream.Read(fromContainer)
			if err != nil {
				t.Fatalf("%d: unexpected error reading container data: %v", i, err)
			}
			if e, a := test.containerData, string(fromContainer[0:n]); e != a {
				t.Fatalf("%d: expected to receive '%v' from container, got '%v'", i, e, a)
			}
		}

		<-portForwardFuncDone
	}
}

func TestCRIHandler(t *testing.T) {
	fw := newServerTest()
	defer fw.testHTTPServer.Close()

	const (
		path  = "/cri/exec/123456abcdef"
		query = "cmd=echo+foo"
	)
	resp, err := http.Get(fw.testHTTPServer.URL + path + "?" + query)
	require.NoError(t, err)
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, "GET", fw.criHandler.RequestReceived.Method)
	assert.Equal(t, path, fw.criHandler.RequestReceived.URL.Path)
	assert.Equal(t, query, fw.criHandler.RequestReceived.URL.RawQuery)
}
