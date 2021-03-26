/*
Copyright 2021 The Kubernetes Authors.

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

package apihelper

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/url"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	k8sclientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var (
	// unsupportedPlatformErr is a sentinel error retuned from NewAPIEndpointHelper when the test is executed on an unsupported platform.
	// An unsupported platform is a platform on which we don't know how to identify all running instances of Kube API servers.
	//
	// we use this error in your test to provide context to the end users, see WaitForAll method
	unsupportedPlatformErr = errors.New("unable to detect all replicas of Kube API server on this platform")
)

type APIHelper struct {
	// clients a slice of clients to individual Kube API servers
	clients []*clientset.Clientset

	// cleanUpFn holds cleanup routine
	cleanUpFn func()

	// unsupportedPlatform a flag used to print the warning log (see WaitForAll method)
	unsupportedPlatform bool
}

// WaitForAll tries a condition func until it returns true, an error, or the timeout is reached against all deployed API servers.
// If this function is executed on a unsupported platform a warning message is logged and it exits.
func (h *APIHelper) WaitForAll(waitTimeout time.Duration, conditionFn func(cs *clientset.Clientset) (bool, error)) error {
	if h.unsupportedPlatform {
		framework.Logf("This test runs with an API server that does not report endpoints to the kubernetes service in the default namespace." + " " +
			"We cannot determine how many control plane API servers are running, which means we can't verify that all instances have stopped serving."  + " " +
			"You can ignore this warning if you are running only a single API server instance.")
		return nil
	}
	return wait.Poll(500*time.Millisecond, waitTimeout, mustSucceedForAllAPIServers(h.clients, conditionFn))
}

// CleanUp executed the cleanup routine.
// This method MUST be called after a successful call to NewAPIEndpointHelper function.
func (h *APIHelper) CleanUp() {
	h.cleanUpFn()
}

// NewAPIEndpointHelper is a convenience function for checking conditions against all deployed Kube API servers.
// Use WaitForAll for checking a condition on all servers and CleanUp for cleaning up after the test.
// Note that at the moment we only support detecting servers by getting API endpoints appropriately.
func NewAPIEndpointHelper(client clientset.Interface, namespace string) (*APIHelper, error) {
	apiHelper := &APIHelper{
		cleanUpFn: func(){/* noop */},
	}
	kubeAPIServersPorts, cleanUpFn, err := setupAPIServersProxyPodAndPortForward(client, namespace)
	if err != nil {
		cleanUpFn()
		if err == unsupportedPlatformErr {
			// suppress the error, we are going to warn the caller when it calls WaitForAll method
			apiHelper.unsupportedPlatform = true
			return apiHelper, nil
		}
		return nil, err
	}
	config, err := framework.LoadConfig()
	if err != nil {
		return nil, err
	}

	apiHelper.cleanUpFn = cleanUpFn
	for _, kubeAPIServerPort := range kubeAPIServersPorts {
		configCopy := *config
		setDefaultServerName(&configCopy)
		configCopy.Host = fmt.Sprintf("https://localhost:%d", kubeAPIServerPort)
		newClient, err := clientset.NewForConfig(&configCopy)
		if err != nil {
			return nil, err
		}
		apiHelper.clients = append(apiHelper.clients, newClient)
	}
	return apiHelper, nil
}

// mustSucceedForAllAPIServers calls f multiple times on success and only returns true if all calls are successful.
// This is necessary to avoid flaking tests where one call might hit a good apiserver while in HA other apiservers
// might be lagging behind.
func mustSucceedForAllAPIServers(kubeClients []*clientset.Clientset, f func(*clientset.Clientset) (bool, error)) func() (bool, error) {
	return func() (bool, error) {
		for i := 0; i < len(kubeClients); i++ {
			ok, err := f(kubeClients[i])
			if err != nil || !ok {
				return ok, err
			}
		}
		return true, nil
	}
}

// setupAPIServersProxyPodAndPortForward a convenience method that creates and runs a pod that proxies connections to the API servers.
// It also uses kubectl port-forward to route local connections to that pod.
func setupAPIServersProxyPodAndPortForward(client clientset.Interface, namespace string) ([]int, func(), error) {
	noopfn := func() {}
	apis, err := getAllAPIServersEndpoint(client)
	if err != nil {
		return nil, noopfn, err
	}
	proxyPod, remotePorts, err := apiServersProxyPod(apis)
	if err != nil {
		return nil, noopfn, err
	}
	if err := createAndWaitForPodRunning(client, namespace, proxyPod); err != nil {
		return nil, noopfn, err
	}
	cmd := runKubectlPortForward(namespace, proxyPod.Name, remotePorts)
	return cmd.localPorts, cmd.stop, nil
}

// portForwardCommand captures running cmd for clean up purposes and a list of local listening ports
type portForwardCommand struct {
	cmd        *exec.Cmd
	localPorts []int
}

// Stop attempts to gracefully stop `kubectl port-forward`, only killing it if necessary.
// This helps avoid spdy goroutine leaks in the Kubelet.
func (c *portForwardCommand) stop() {
	// SIGINT signals that kubectl port-forward should gracefully terminate
	if err := c.cmd.Process.Signal(syscall.SIGINT); err != nil {
		framework.Logf("error sending SIGINT to kubectl port-forward: %v", err)
	}

	// try to wait for a clean exit
	done := make(chan error)
	go func() {
		done <- c.cmd.Wait()
	}()

	expired := time.NewTimer(wait.ForeverTestTimeout)
	defer expired.Stop()

	select {
	case err := <-done:
		if err == nil {
			// success
			return
		}
		framework.Logf("error waiting for kubectl port-forward to exit: %v", err)
	case <-expired.C:
		framework.Logf("timed out waiting for kubectl port-forward to exit")
	}

	framework.Logf("trying to forcibly kill kubectl port-forward")
	framework.TryKill(c.cmd)
}

// RunKubectlPortForward runs port-forward via kubectl on multiple ports - warning, this may need root functionality on some systems.
func runKubectlPortForward(namespace, podName string, remotePorts []int) *portForwardCommand {
	remotePortsToStrArr := func(remotePorts []int) []string {
		ret := []string{}
		for _, port := range remotePorts {
			ret = append(ret, fmt.Sprintf(":%d", port))
		}
		return ret
	}

	args := []string{"port-forward", fmt.Sprintf("--namespace=%v", namespace), podName}
	args = append(args, remotePortsToStrArr(remotePorts)...)

	tk := e2ekubectl.NewTestKubeconfig(framework.TestContext.CertDir, framework.TestContext.Host, framework.TestContext.KubeConfig, framework.TestContext.KubeContext, framework.TestContext.KubectlPath, namespace)
	cmd := tk.KubectlCmd(args...)

	// This is somewhat ugly but is the only way to retrieve the port that was picked
	// by the port-forward command. We don't want to hard code the port as we have no
	// way of guaranteeing we can pick one that isn't in use, particularly on Jenkins.
	framework.Logf("starting kubectl port-forward command and streaming output")
	cmdstdout, _, err := framework.StartCmdAndStreamOutput(cmd)
	if err != nil {
		framework.Failf("Failed to start port-forward command: %v", err)
	}

	var localPorts []int
	err = wait.Poll(500*time.Millisecond, 10*time.Second, func() (done bool, err error) {
		buf := make([]byte, 128*len(remotePorts))
		var n int
		framework.Logf("reading from `kubectl port-forward` command's stdout")
		if n, err = cmdstdout.Read(buf); err != nil {
			return false, fmt.Errorf("failed to read from kubectl port-forward stdout: %v", err)
		}
		portForwardOutput := string(buf[:n])

		localPorts = make([]int, len(remotePorts))
		for index, remotePort := range remotePorts {
			portForwardExpr := fmt.Sprintf("Forwarding from (127.0.0.1|\\[::1\\]):([0-9]+) -> %d", remotePort)
			portForwardRegexp, err := regexp.Compile(portForwardExpr)
			if err != nil {
				return false, fmt.Errorf("failed to compile a regexp for finding a local listening port, err: %v, expression: %s", err, portForwardExpr)
			}
			match := portForwardRegexp.FindStringSubmatch(portForwardOutput)
			if len(match) != 3 {
				framework.Logf("failed to parse kubectl port-forward output: %q, with %q reg exp, expected to find exactly 3 matches, found: %v", portForwardOutput, portForwardExpr, match)
				return false, nil
			}

			localListenPort, err := strconv.Atoi(match[2])
			if err != nil {
				return false, fmt.Errorf("error converting %s to an int: %v", match[2], err)
			}
			localPorts[index] = localListenPort
		}
		if len(localPorts) != len(remotePorts) {
			framework.Logf("not all ports have been forwarded, found %d forwarded ports but want %d", len(localPorts), len(remotePorts))
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		framework.Failf("Timeout waiting/parsing kubectl port-forward command's output: %v", err)
	}

	return &portForwardCommand{
		cmd:        cmd,
		localPorts: localPorts,
	}
}

// apiServersProxyPod creates a pod that once run proxies connections to the given apiServers.
// It also returns a remote ports list for convenience that will be used by kubectl port-forward
func apiServersProxyPod(apiServers []string) (*v1.Pod, []int, error) {
	script := `
       apk add socat
       socat TCP-LISTEN:${LOCAL_LISTEN_PORT},fork TCP:${SERVER_IP}:${SERVER_PORT}
`

	proxyListenPorts := []int{}
	proxyListenPort := 8443
	containers := []v1.Container{}
	for index, apiServer := range apiServers {
		ipPort := strings.Split(apiServer, ":")
		if len(ipPort) != 2 {
			return nil, nil, fmt.Errorf("incorrect apiServer =%s, expected to find an IP address and port in the form of IP:PORT", apiServer)
		}

		r := strings.NewReplacer(
			"${LOCAL_LISTEN_PORT}", fmt.Sprintf("%d", proxyListenPort),
			"${SERVER_IP}", ipPort[0],
			"${SERVER_PORT}", ipPort[1],
		)
		modifiedScript := r.Replace(script)

		containers = append(containers, v1.Container{
			Name: fmt.Sprintf("server-%d", index),
			Ports: []v1.ContainerPort{
				{ContainerPort: int32(proxyListenPort)},
			},
			Image:   imageutils.GetE2EImage(imageutils.Agnhost),
			Command: []string{"/bin/sh", "-c"},
			Args:    []string{modifiedScript},
		})
		proxyListenPorts = append(proxyListenPorts, proxyListenPort)
		proxyListenPort++
	}

	name := "api-servers-proxy"
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name + "-" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			Containers: containers,
		},
	}, proxyListenPorts, nil
}

func getAllAPIServersEndpoint(c k8sclientset.Interface) ([]string, error) {
	eps, err := c.CoreV1().Endpoints(metav1.NamespaceDefault).Get(context.TODO(), "kubernetes", metav1.GetOptions{})
	if err != nil && apierrors.IsNotFound(err) {
		return nil, unsupportedPlatformErr
	}

	apiServers := []string{}
	for _, s := range eps.Subsets {
		var port int32
		for _, p := range s.Ports {
			if p.Name == "https" {
				port = p.Port
				break
			}
		}
		if port == 0 {
			continue
		}
		for _, ep := range s.Addresses {
			apiServers = append(apiServers, fmt.Sprintf("%s:%d", ep.IP, port))
		}
		break
	}
	if len(apiServers) == 0 {
		return nil, unsupportedPlatformErr
	}
	return apiServers, nil
}

func createAndWaitForPodRunning(client clientset.Interface, namespace string, pod *v1.Pod) error {
	createdPod, err := client.CoreV1().Pods(namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		return err
	}
	if err = e2epod.WaitForPodRunningInNamespace(client, createdPod); err != nil {
		framework.Failf("Pod %v did not start running: %v", pod.Name, err)
	}
	return nil
}

// setDefaultServerName extract the hostname from the config.Host and sets it in config.ServerName
// the ServerName is passed to the server for SNI and is used in the client to check server certificates.
//
// note:
// if the ServerName has been already specified calling this method has no effect
func setDefaultServerName(config *rest.Config) error {
	if len(config.ServerName) > 0 {
		return nil
	}
	u, err := url.Parse(config.Host)
	if err != nil {
		return err
	}
	host, _, err := net.SplitHostPort(u.Host)
	if err != nil {
		// assume u.Host contains only host portion
		config.ServerName = u.Host
		return nil
	}
	config.ServerName = host
	return nil
}
