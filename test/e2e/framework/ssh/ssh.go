/*
Copyright 2018 The Kubernetes Authors.

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

package ssh

import (
	"bytes"
	"context"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/onsi/gomega"

	"golang.org/x/crypto/ssh"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

const (
	// SSHPort is tcp port number of SSH
	SSHPort = "22"

	// pollNodeInterval is how often to Poll pods.
	pollNodeInterval = 2 * time.Second

	// singleCallTimeout is how long to try single API calls (like 'get' or 'list'). Used to prevent
	// transient failures from failing tests.
	singleCallTimeout = 5 * time.Minute

	// sshBastionEnvKey is the environment variable key for running SSH commands via bastion.
	sshBastionEnvKey = "KUBE_SSH_BASTION"
)

// GetSigner returns an ssh.Signer for the provider ("gce", etc.) that can be
// used to SSH to their nodes.
func GetSigner(provider string) (ssh.Signer, error) {
	// honor a consistent SSH key across all providers
	if path := os.Getenv("KUBE_SSH_KEY_PATH"); len(path) > 0 {
		return makePrivateKeySignerFromFile(path)
	}

	// Select the key itself to use. When implementing more providers here,
	// please also add them to any SSH tests that are disabled because of signer
	// support.
	keyfile := ""
	switch provider {
	case "gce", "gke", "kubemark":
		keyfile = os.Getenv("GCE_SSH_KEY")
		if keyfile == "" {
			keyfile = "google_compute_engine"
		}
	case "aws", "eks":
		keyfile = os.Getenv("AWS_SSH_KEY")
		if keyfile == "" {
			keyfile = "kube_aws_rsa"
		}
	case "local", "vsphere":
		keyfile = os.Getenv("LOCAL_SSH_KEY")
		if keyfile == "" {
			keyfile = "id_rsa"
		}
	case "skeleton":
		keyfile = os.Getenv("KUBE_SSH_KEY")
		if keyfile == "" {
			keyfile = "id_rsa"
		}
	default:
		return nil, fmt.Errorf("GetSigner(...) not implemented for %s", provider)
	}

	// Respect absolute paths for keys given by user, fallback to assuming
	// relative paths are in ~/.ssh
	if !filepath.IsAbs(keyfile) {
		keydir := filepath.Join(os.Getenv("HOME"), ".ssh")
		keyfile = filepath.Join(keydir, keyfile)
	}

	return makePrivateKeySignerFromFile(keyfile)
}

func makePrivateKeySignerFromFile(key string) (ssh.Signer, error) {
	buffer, err := ioutil.ReadFile(key)
	if err != nil {
		return nil, fmt.Errorf("error reading SSH key %s: '%v'", key, err)
	}

	signer, err := ssh.ParsePrivateKey(buffer)
	if err != nil {
		return nil, fmt.Errorf("error parsing SSH key: '%v'", err)
	}

	return signer, err
}

// NodeSSHHosts returns SSH-able host names for all schedulable nodes.
// If it can't find any external IPs, it falls back to
// looking for internal IPs. If it can't find an internal IP for every node it
// returns an error, though it still returns all hosts that it found in that
// case.
func NodeSSHHosts(c clientset.Interface) ([]string, error) {
	nodelist := waitListSchedulableNodesOrDie(c)

	hosts := nodeAddresses(nodelist, v1.NodeExternalIP)
	// If  ExternalIPs aren't available for all nodes, try falling back to the InternalIPs.
	if len(hosts) < len(nodelist.Items) {
		e2elog.Logf("No external IP address on nodes, falling back to internal IPs")
		hosts = nodeAddresses(nodelist, v1.NodeInternalIP)
	}

	// Error if neither External nor Internal IPs weren't available for all nodes.
	if len(hosts) != len(nodelist.Items) {
		return hosts, fmt.Errorf(
			"only found %d IPs on nodes, but found %d nodes. Nodelist: %v",
			len(hosts), len(nodelist.Items), nodelist)
	}

	lenHosts := len(hosts)
	wg := &sync.WaitGroup{}
	wg.Add(lenHosts)
	sshHosts := make([]string, 0, lenHosts)
	var sshHostsLock sync.Mutex

	for _, host := range hosts {
		go func(host string) {
			defer wg.Done()
			if canConnect(host) {
				e2elog.Logf("Assuming SSH on host %s", host)
				sshHostsLock.Lock()
				sshHosts = append(sshHosts, net.JoinHostPort(host, SSHPort))
				sshHostsLock.Unlock()
			} else {
				e2elog.Logf("Skipping host %s because it does not run anything on port %s", host, SSHPort)
			}
		}(host)
	}
	wg.Wait()

	return sshHosts, nil
}

// canConnect returns true if a network connection is possible to the SSHPort.
func canConnect(host string) bool {
	if _, ok := os.LookupEnv(sshBastionEnvKey); ok {
		return true
	}
	hostPort := net.JoinHostPort(host, SSHPort)
	conn, err := net.DialTimeout("tcp", hostPort, 3*time.Second)
	if err != nil {
		e2elog.Logf("cannot dial %s: %v", hostPort, err)
		return false
	}
	conn.Close()
	return true
}

// Result holds the execution result of SSH command
type Result struct {
	User   string
	Host   string
	Cmd    string
	Stdout string
	Stderr string
	Code   int
}

// NodeExec execs the given cmd on node via SSH. Note that the nodeName is an sshable name,
// eg: the name returned by framework.GetMasterHost(). This is also not guaranteed to work across
// cloud providers since it involves ssh.
func NodeExec(nodeName, cmd, provider string) (Result, error) {
	return SSH(cmd, net.JoinHostPort(nodeName, SSHPort), provider)
}

// SSH synchronously SSHs to a node running on provider and runs cmd. If there
// is no error performing the SSH, the stdout, stderr, and exit code are
// returned.
func SSH(cmd, host, provider string) (Result, error) {
	result := Result{Host: host, Cmd: cmd}

	// Get a signer for the provider.
	signer, err := GetSigner(provider)
	if err != nil {
		return result, fmt.Errorf("error getting signer for provider %s: '%v'", provider, err)
	}

	// RunSSHCommand will default to Getenv("USER") if user == "", but we're
	// defaulting here as well for logging clarity.
	result.User = os.Getenv("KUBE_SSH_USER")
	if result.User == "" {
		result.User = os.Getenv("USER")
	}

	if bastion := os.Getenv(sshBastionEnvKey); len(bastion) > 0 {
		stdout, stderr, code, err := runSSHCommandViaBastion(cmd, result.User, bastion, host, signer)
		result.Stdout = stdout
		result.Stderr = stderr
		result.Code = code
		return result, err
	}

	stdout, stderr, code, err := runSSHCommand(cmd, result.User, host, signer)
	result.Stdout = stdout
	result.Stderr = stderr
	result.Code = code

	return result, err
}

// runSSHCommandViaBastion returns the stdout, stderr, and exit code from running cmd on
// host as specific user, along with any SSH-level error.
func runSSHCommand(cmd, user, host string, signer ssh.Signer) (string, string, int, error) {
	if user == "" {
		user = os.Getenv("USER")
	}
	// Setup the config, dial the server, and open a session.
	config := &ssh.ClientConfig{
		User:            user,
		Auth:            []ssh.AuthMethod{ssh.PublicKeys(signer)},
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
	}
	client, err := ssh.Dial("tcp", host, config)
	if err != nil {
		err = wait.Poll(5*time.Second, 20*time.Second, func() (bool, error) {
			fmt.Printf("error dialing %s@%s: '%v', retrying\n", user, host, err)
			if client, err = ssh.Dial("tcp", host, config); err != nil {
				return false, err
			}
			return true, nil
		})
	}
	if err != nil {
		return "", "", 0, fmt.Errorf("error getting SSH client to %s@%s: '%v'", user, host, err)
	}
	defer client.Close()
	session, err := client.NewSession()
	if err != nil {
		return "", "", 0, fmt.Errorf("error creating session to %s@%s: '%v'", user, host, err)
	}
	defer session.Close()

	// Run the command.
	code := 0
	var bout, berr bytes.Buffer
	session.Stdout, session.Stderr = &bout, &berr
	if err = session.Run(cmd); err != nil {
		// Check whether the command failed to run or didn't complete.
		if exiterr, ok := err.(*ssh.ExitError); ok {
			// If we got an ExitError and the exit code is nonzero, we'll
			// consider the SSH itself successful (just that the command run
			// errored on the host).
			if code = exiterr.ExitStatus(); code != 0 {
				err = nil
			}
		} else {
			// Some other kind of error happened (e.g. an IOError); consider the
			// SSH unsuccessful.
			err = fmt.Errorf("failed running `%s` on %s@%s: '%v'", cmd, user, host, err)
		}
	}
	return bout.String(), berr.String(), code, err
}

// runSSHCommandViaBastion returns the stdout, stderr, and exit code from running cmd on
// host as specific user, along with any SSH-level error. It uses an SSH proxy to connect
// to bastion, then via that tunnel connects to the remote host. Similar to
// sshutil.RunSSHCommand but scoped to the needs of the test infrastructure.
func runSSHCommandViaBastion(cmd, user, bastion, host string, signer ssh.Signer) (string, string, int, error) {
	// Setup the config, dial the server, and open a session.
	config := &ssh.ClientConfig{
		User:            user,
		Auth:            []ssh.AuthMethod{ssh.PublicKeys(signer)},
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
		Timeout:         150 * time.Second,
	}
	bastionClient, err := ssh.Dial("tcp", bastion, config)
	if err != nil {
		err = wait.Poll(5*time.Second, 20*time.Second, func() (bool, error) {
			fmt.Printf("error dialing %s@%s: '%v', retrying\n", user, bastion, err)
			if bastionClient, err = ssh.Dial("tcp", bastion, config); err != nil {
				return false, err
			}
			return true, nil
		})
	}
	if err != nil {
		return "", "", 0, fmt.Errorf("error getting SSH client to %s@%s: %v", user, bastion, err)
	}
	defer bastionClient.Close()

	conn, err := bastionClient.Dial("tcp", host)
	if err != nil {
		return "", "", 0, fmt.Errorf("error dialing %s from bastion: %v", host, err)
	}
	defer conn.Close()

	ncc, chans, reqs, err := ssh.NewClientConn(conn, host, config)
	if err != nil {
		return "", "", 0, fmt.Errorf("error creating forwarding connection %s from bastion: %v", host, err)
	}
	client := ssh.NewClient(ncc, chans, reqs)
	defer client.Close()

	session, err := client.NewSession()
	if err != nil {
		return "", "", 0, fmt.Errorf("error creating session to %s@%s from bastion: '%v'", user, host, err)
	}
	defer session.Close()

	// Run the command.
	code := 0
	var bout, berr bytes.Buffer
	session.Stdout, session.Stderr = &bout, &berr
	if err = session.Run(cmd); err != nil {
		// Check whether the command failed to run or didn't complete.
		if exiterr, ok := err.(*ssh.ExitError); ok {
			// If we got an ExitError and the exit code is nonzero, we'll
			// consider the SSH itself successful (just that the command run
			// errored on the host).
			if code = exiterr.ExitStatus(); code != 0 {
				err = nil
			}
		} else {
			// Some other kind of error happened (e.g. an IOError); consider the
			// SSH unsuccessful.
			err = fmt.Errorf("failed running `%s` on %s@%s: '%v'", cmd, user, host, err)
		}
	}
	return bout.String(), berr.String(), code, err
}

// LogResult records result log
func LogResult(result Result) {
	remote := fmt.Sprintf("%s@%s", result.User, result.Host)
	e2elog.Logf("ssh %s: command:   %s", remote, result.Cmd)
	e2elog.Logf("ssh %s: stdout:    %q", remote, result.Stdout)
	e2elog.Logf("ssh %s: stderr:    %q", remote, result.Stderr)
	e2elog.Logf("ssh %s: exit code: %d", remote, result.Code)
}

// IssueSSHCommandWithResult tries to execute a SSH command and returns the execution result
func IssueSSHCommandWithResult(cmd, provider string, node *v1.Node) (*Result, error) {
	e2elog.Logf("Getting external IP address for %s", node.Name)
	host := ""
	for _, a := range node.Status.Addresses {
		if a.Type == v1.NodeExternalIP && a.Address != "" {
			host = net.JoinHostPort(a.Address, SSHPort)
			break
		}
	}

	if host == "" {
		// No external IPs were found, let's try to use internal as plan B
		for _, a := range node.Status.Addresses {
			if a.Type == v1.NodeInternalIP && a.Address != "" {
				host = net.JoinHostPort(a.Address, SSHPort)
				break
			}
		}
	}

	if host == "" {
		return nil, fmt.Errorf("couldn't find any IP address for node %s", node.Name)
	}

	e2elog.Logf("SSH %q on %s(%s)", cmd, node.Name, host)
	result, err := SSH(cmd, host, provider)
	LogResult(result)

	if result.Code != 0 || err != nil {
		return nil, fmt.Errorf("failed running %q: %v (exit code %d, stderr %v)",
			cmd, err, result.Code, result.Stderr)
	}

	return &result, nil
}

// IssueSSHCommand tries to execute a SSH command
func IssueSSHCommand(cmd, provider string, node *v1.Node) error {
	_, err := IssueSSHCommandWithResult(cmd, provider, node)
	if err != nil {
		return err
	}
	return nil
}

// nodeAddresses returns the first address of the given type of each node.
func nodeAddresses(nodelist *v1.NodeList, addrType v1.NodeAddressType) []string {
	hosts := []string{}
	for _, n := range nodelist.Items {
		for _, addr := range n.Status.Addresses {
			if addr.Type == addrType && addr.Address != "" {
				hosts = append(hosts, addr.Address)
				break
			}
		}
	}
	return hosts
}

// waitListSchedulableNodes is a wrapper around listing nodes supporting retries.
func waitListSchedulableNodes(c clientset.Interface) (*v1.NodeList, error) {
	var nodes *v1.NodeList
	var err error
	if wait.PollImmediate(pollNodeInterval, singleCallTimeout, func() (bool, error) {
		nodes, err = c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector().String()})
		if err != nil {
			return false, err
		}
		return true, nil
	}) != nil {
		return nodes, err
	}
	return nodes, nil
}

// waitListSchedulableNodesOrDie is a wrapper around listing nodes supporting retries.
func waitListSchedulableNodesOrDie(c clientset.Interface) *v1.NodeList {
	nodes, err := waitListSchedulableNodes(c)
	if err != nil {
		expectNoError(err, "Non-retryable failure or timed out while listing nodes for e2e cluster.")
	}
	return nodes
}

// expectNoError checks if "err" is set, and if so, fails assertion while logging the error.
func expectNoError(err error, explain ...interface{}) {
	expectNoErrorWithOffset(1, err, explain...)
}

// expectNoErrorWithOffset checks if "err" is set, and if so, fails assertion while logging the error at "offset" levels above its caller
// (for example, for call chain f -> g -> ExpectNoErrorWithOffset(1, ...) error would be logged for "f").
func expectNoErrorWithOffset(offset int, err error, explain ...interface{}) {
	if err != nil {
		e2elog.Logf("Unexpected error occurred: %v", err)
	}
	gomega.ExpectWithOffset(1+offset, err).NotTo(gomega.HaveOccurred(), explain...)
}
