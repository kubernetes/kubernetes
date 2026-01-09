/*
Copyright 2025 The Kubernetes Authors.

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

// Package localupcluster contains wrapper code around invoking hack/local-up-cluster.sh
// and managing the resulting cluster.
//
// The basic mode of operation is:
//   - local-up-cluster.sh contains all the logic of how to configure and invoke components.
//   - local-up-cluster.sh is run in dry-run mode, which prints all commands and their parameters
//     without actually running them, except for etcd: etcd's lifecycle is managed as before
//     by the caller or local-up-cluster.sh.
//   - local-up-cluster.sh is kept running as long as the cluster runs to keep etcd running and
//     generated configuration files around.
//   - This package takes care of running the commands, including output redirection.
//   - It can stop commands and run them again using different binaries to similar upgrades
//     or downgrades.
package localupcluster

import (
	"context"
	"crypto/tls"
	"fmt"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type KubeComponentName string

// Component names.
//
// They match the names in the local-up-cluster.sh output, if the script runs those components.
const (
	KubeAPIServer         = KubeComponentName("kube-apiserver")
	KubeControllerManager = KubeComponentName("kube-controller-manager")
	KubeScheduler         = KubeComponentName("kube-scheduler")
	Kubelet               = KubeComponentName("kubelet")
	KubeProxy             = KubeComponentName("kube-proxy")
	Kubectl               = KubeComponentName("kubectl")
	LocalUpCluster        = KubeComponentName("local-up-cluster")
)

// Kubernetes components running in the cluster, in the order in which they need to be started and upgraded.
var KubeClusterComponents = []KubeComponentName{KubeAPIServer, KubeControllerManager, KubeScheduler, Kubelet, KubeProxy}

// RUN <name> <command line> in the local-up-cluster.sh output marks commands that we need to run.
const localUpClusterRunPrefix = "RUN "

func repoRoot(tCtx ktesting.TContext) string {
	for i := 0; ; i++ {
		dir := path.Join(".", strings.Repeat("../", i))
		_, err := os.Stat(dir)
		tCtx.ExpectNoError(err, "examine parent directory while looking for hack/local-up-cluster.sh (not invoked inside the Kubernetes repository?)")
		_, err = os.Stat(path.Join(dir, "hack/local-up-cluster.sh"))
		if err == nil {
			dir, err = filepath.Abs(dir)
			tCtx.ExpectNoError(err, "turn into absolute path")
			return dir
		}
	}
}

func New(tCtx ktesting.TContext) *Cluster {
	tCtx.Helper()
	c := &Cluster{}
	tCtx.CleanupCtx(c.Stop)
	return c
}

// Cluster represents one cluster.
//
// hack/local-up-cluster.sh must be functional in the current environment. If necessary,
// env variables like CONTAINER_RUNTIME_ENDPOINT have to be set to adapt the script
// to the host system. The current user must have permission to use the container runtime.
// All Kubernetes components run as that user with files stored in a test temp directory.
//
// local-up-cluster.sh does not support more than one cluster per host, so
// tests using this package have to run sequentially.
type Cluster struct {
	running    map[KubeComponentName]*Cmd
	dir        string
	kubeConfig string
	settings   map[string]string
}

// Start brings up the cluster anew. If it was already running, it will be stopped first.
//
// The cluster will be stopped automatically at the end of the test.
// If the ARTIFACTS env variable is set and the test failed,
// log files of the kind cluster get dumped into
// $ARTIFACTS/<test name>/kind/<cluster name> before stopping it.
//
// The binary directory must contain kube-apiserver, kube-controller-manager,
// kube-scheduler, kube-proxy, and kubelet. Those binaries can be from a previous
// Kubernetes release. They will be invoked with parameters  as defined in the
// *current* local-up-cluster.sh. This works as long as local-up-cluster.sh in its
// default configuration doesn't depend on something which was added only recently.
func (c *Cluster) Start(tCtx ktesting.TContext, bindir string, localUpClusterEnv map[string]string) {
	tCtx.Helper()
	c.Stop(tCtx)
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		// Intentional additional lambda function for source code location in log output.
		c.Stop(tCtx)
	})

	if artifacts, ok := os.LookupEnv("ARTIFACTS"); ok {
		// Sanitize the name:
		// - remove E2E [] tags
		// - replace whitespaces and some special characters with hyphens
		testName := tCtx.Name()
		testName = regexp.MustCompile(`\s*\[[^\]]*\]`).ReplaceAllString(testName, "")
		testName = regexp.MustCompile(`[[:space:]/:()\\]+`).ReplaceAllString(testName, "-")
		testName = strings.Trim(testName, "-")
		c.dir = path.Join(artifacts, testName)
		tCtx.ExpectNoError(os.MkdirAll(c.dir, 0766), "create artifacts directory for test")
	} else {
		c.dir = tCtx.TempDir()
	}
	c.running = make(map[KubeComponentName]*Cmd)
	c.settings = make(map[string]string)

	// Spawn local-up-cluster.sh in background, keep it running (for etcd!),
	// parse output to pick up commands and run them in order.
	lines := make(chan Output, 100)
	cmd := &Cmd{
		Name: string(LocalUpCluster),
		CommandLine: []string{
			path.Join(repoRoot(tCtx), "hack/local-up-cluster.sh"),
			"-o", bindir,
			"-d", // dry run
		},
		ProcessOutput: func(output Output) {
			// Redirect processing into the main goroutine.
			lines <- output
		},
		AdditionalEnv: localUpClusterEnv,
	}

	kubeVerboseStr := cmd.AdditionalEnv["KUBE_VERBOSE"]
	kubeVerboseVal, err := strconv.Atoi(kubeVerboseStr)
	if kubeVerboseStr == "" {
		kubeVerboseVal = 0
	} else {
		tCtx.ExpectNoError(err, "KUBE_VERBOSE")
	}
	if kubeVerboseVal < 2 {
		cmd.AdditionalEnv["KUBE_VERBOSE"] = "2" // Enables -x for configuration variable assignments.
	}
	cmd.Start(tCtx)
	c.running[LocalUpCluster] = cmd

processLocalUpClusterOutput:
	for {
		select {
		case <-tCtx.Done():
			c.Stop(tCtx)
			tCtx.Fatalf("interrupted cluster startup: %w", context.Cause(tCtx))
		case output := <-lines:
			if c.processLocalUpClusterOutput(tCtx, output) {
				break processLocalUpClusterOutput
			}
		}
	}
	tCtx.Logf("cluster is running, use KUBECONFIG=%s to access it", c.kubeConfig)
}

// Matches e.g. "+ API_SECURE_PORT=6443".
var varAssignment = regexp.MustCompile(`^\+ ([A-Z0-9_]+)=(.*)$`)

func (c *Cluster) processLocalUpClusterOutput(tCtx ktesting.TContext, output Output) bool {
	if output.EOF {
		if output.Line != "" {
			tCtx.Fatalf("%s output processing failed: %s", LocalUpCluster, output.Line)
		}
		tCtx.Fatalf("%s terminated unexpectedly", LocalUpCluster)
	}

	tCtx.Logf("local-up-cluster: %s", output.Line)

	if strings.HasPrefix(output.Line, localUpClusterRunPrefix) {
		line := output.Line[len(localUpClusterRunPrefix):]
		parts := strings.SplitN(line, ": ", 2)
		if len(parts) != 2 {
			tCtx.Fatalf("unexpected RUN line: %s", output.Line)
		}
		name := parts[0]
		cmdLine := parts[1]

		// Cluster components are kept running.
		if slices.Contains(KubeClusterComponents, KubeComponentName(name)) {
			c.runKubeComponent(tCtx, KubeComponentName(name), cmdLine)
			return false
		}

		// Other commands get invoked and need to terminate before we proceed.
		c.runCmd(tCtx, name, cmdLine)
		return false
	}
	if m := varAssignment.FindStringSubmatch(output.Line); m != nil {
		c.settings[m[1]] = m[2]
		if m[1] == "CERT_DIR" {
			c.kubeConfig = path.Join(m[2], "admin.kubeconfig")
		}
		return false
	}
	if strings.Contains(output.Line, "Local etcd is running. Run commands.") {
		// We have seen and processed all commands.
		// Time to start testing...
		return true
	}

	return false
}

func (c *Cluster) runKubeComponent(tCtx ktesting.TContext, component KubeComponentName, command string) {
	commandLine := fromLocalUpClusterOutput(command)

	cmd := &Cmd{
		Name:        string(component),
		CommandLine: commandLine,
		// Number gets bumped when restarting.
		LogFile: path.Join(c.dir, fmt.Sprintf("%s-0.log", component)),
		// Stopped via Cluster.Stop.
		KeepRunning: true,
	}

	c.runComponentWithRetry(tCtx, cmd)
}

func (c *Cluster) runCmd(tCtx ktesting.TContext, name, command string) {
	commandLine := fromLocalUpClusterOutput(command)
	cmd := &Cmd{
		Name:         name,
		CommandLine:  commandLine,
		GatherOutput: true,
	}
	cmd.Start(tCtx)
	cmd.Wait(tCtx)
}

func fromLocalUpClusterOutput(command string) []string {
	// The assumption here is that arguments don't contain spaces.
	// We cannot pass the entire string to a shell and let the shell do
	// the parsing because some arguments contain special characters like $
	// without quoting them.
	return strings.Split(command, " ")
}

// Stop ensures that the cluster is not running anymore.
func (c *Cluster) Stop(tCtx ktesting.TContext) {
	tCtx.Helper()
	for _, component := range slices.Backward(KubeClusterComponents) {
		cmd := c.running[component]
		if cmd == nil {
			continue
		}
		tCtx.Logf("stopping %s", component)
		cmd.Stop(tCtx, "stopping cluster")
	}
}

// LoadConfig returns the REST config for the running cluster.
func (c *Cluster) LoadConfig(tCtx ktesting.TContext) *restclient.Config {
	tCtx.Helper()
	cfg, err := clientcmd.LoadFromFile(c.kubeConfig)
	tCtx.ExpectNoError(err, "load KubeConfig")
	config, err := clientcmd.NewDefaultClientConfig(*cfg, nil).ClientConfig()
	tCtx.ExpectNoError(err, "construct REST config")
	return config

}

// GetSystemLogs returns the output of the given component, the empty string and false if not started.
func (c *Cluster) GetSystemLogs(tCtx ktesting.TContext, component KubeComponentName) (string, bool) {
	cmd, ok := c.running[component]
	if !ok {
		return "", false
	}
	return cmd.Output(tCtx), true
}

type ModifyOptions struct {
	// BinDir specifies where to find the replacement Kubernetes components.
	// If empty, then only components explicitly listed in FileByComponent
	// get modified.
	BinDir string

	// FileByComponent overrides BinDir for those components which are specified here.
	FileByComponent map[KubeComponentName]string

	// Upgrade determines whether the apiserver gets updated first (upgrade)
	// or last (downgrade).
	Upgrade bool
}

func (m ModifyOptions) GetComponentFile(component KubeComponentName) string {
	if file, ok := m.FileByComponent[component]; ok {
		return file
	}
	if m.BinDir != "" {
		return path.Join(m.BinDir, string(component))
	}
	return ""
}

// Modify changes the cluster as described in the options.
// It returns options that can be passed to Modify unchanged
// to restore the original state.
func (c *Cluster) Modify(tCtx ktesting.TContext, options ModifyOptions) ModifyOptions {
	tCtx.Helper()

	restore := ModifyOptions{
		FileByComponent: make(map[KubeComponentName]string),
	}

	restore.Upgrade = !options.Upgrade
	components := slices.Clone(KubeClusterComponents)
	if !options.Upgrade {
		slices.Reverse(components)
	}
	for _, component := range components {
		c.modifyComponent(tCtx, options, component, &restore)
	}
	return restore
}

func (c *Cluster) modifyComponent(tCtx ktesting.TContext, options ModifyOptions, component KubeComponentName, restore *ModifyOptions) {
	tCtx.Helper()
	tCtx = ktesting.Begin(tCtx, fmt.Sprintf("modify %s", component))
	defer ktesting.End(tCtx)

	// We could also do things like turning feature gates on or off.
	// For now we only support replacing the file.
	if fileName := options.GetComponentFile(component); fileName != "" {
		cmd, ok := c.running[component]
		if !ok {
			tCtx.Fatal("not running")
		}
		cmd.Stop(tCtx, "modifying the component")
		delete(c.running, component)

		// Find the command (might be wrapped by sudo!).
		cmdLine := slices.Clone(cmd.CommandLine)
		found := false
		for i := range cmdLine {
			if path.Base(cmdLine[i]) == string(component) {
				found = true
				restore.FileByComponent[component] = cmdLine[i]
				cmdLine[i] = fileName
				break
			}
		}
		if !found {
			tCtx.Fatal("binary filename not found")
		}
		cmd.CommandLine = cmdLine

		// New log file.
		m := regexp.MustCompile(`^(.*)-([[:digit:]]+)\.log$`).FindStringSubmatch(cmd.LogFile)
		if m == nil {
			tCtx.Fatalf("unexpected log file, should have contained number: %s", cmd.LogFile)
		}
		logNum, _ := strconv.Atoi(m[2])
		cmd.LogFile = fmt.Sprintf("%s-%d.log", m[1], logNum+1)

		c.runComponentWithRetry(tCtx, cmd)
	}
}

func (c *Cluster) runComponentWithRetry(tCtx ktesting.TContext, cmd *Cmd) {
	// Sometimes components fail to come up. We have to retry.
	//
	// For example, the apiserver's port might not be free again yet (no SO_LINGER!).
	// Or kube-controller-manager:
	//  I0630 13:20:45.046709   61710 serving.go:380] Generated self-signed cert (/var/run/kubernetes/kube-controller-manager.crt, /var/run/kubernetes/kube-controller-manager.key)
	//  W0630 13:20:45.410578   61710 requestheader_controller.go:204] Unable to get configmap/extension-apiserver-authentication in kube-system.  Usually fixed by 'kubectl create rolebinding -n kube-system ROLEBINDING_NAME --role=extension-apiserver-authentication-reader --serviceaccount=YOUR_NS:YOUR_SA'
	//  E0630 13:20:45.410618   61710 run.go:72] "command failed" err="unable to load configmap based request-header-client-ca-file: configmaps \"extension-apiserver-authentication\" is forbidden: User \"system:kube-controller-manager\" cannot get resource \"configmaps\" in API group \"\" in the namespace \"kube-system\""
	// The kube-controller-manager then is stuck. Perhaps it should retry instead?
	for i := 0; ; i++ {
		tCtx.Logf("running %s with output redirected to %s", cmd.Name, cmd.LogFile)
		cmd.Start(tCtx)
		c.running[KubeComponentName(cmd.Name)] = cmd
		err := func() (finalErr error) {
			tCtx, finalize := ktesting.WithError(tCtx, &finalErr)
			defer finalize()
			c.checkReadiness(tCtx, cmd)
			return nil
		}()
		if err == nil {
			break
		}
		if !cmd.Running() && i < 10 {
			// Retry.
			time.Sleep(time.Second)
			continue
		}
		// Re-raise the failure.
		tCtx.ExpectNoError(err)
	}
}

func (c *Cluster) checkReadiness(tCtx ktesting.TContext, cmd *Cmd) {
	restConfig := c.LoadConfig(tCtx)
	tCtx = ktesting.WithRESTConfig(tCtx, restConfig)
	tCtx = ktesting.Begin(tCtx, fmt.Sprintf("wait for %s readiness", cmd.Name))
	defer ktesting.End(tCtx)

	switch KubeComponentName(cmd.Name) {
	case KubeAPIServer:
		c.checkHealthz(tCtx, cmd, "https", c.settings["API_HOST_IP"], c.settings["API_SECURE_PORT"])
	case KubeScheduler:
		c.checkHealthz(tCtx, cmd, "https", c.settings["API_HOST_IP"], c.settings["SCHEDULER_SECURE_PORT"])
	case KubeControllerManager:
		c.checkHealthz(tCtx, cmd, "https", c.settings["API_HOST_IP"], c.settings["KCM_SECURE_PORT"])
	case KubeProxy:
		c.checkHealthz(tCtx, cmd, "http" /* not an error! */, c.settings["API_HOST_IP"], c.settings["PROXY_HEALTHZ_PORT"])
	case Kubelet:
		c.checkHealthz(tCtx, cmd, "https", c.settings["KUBELET_HOST"], c.settings["KUBELET_PORT"])

		// Also wait for the node to be ready.
		tCtx = ktesting.Begin(tCtx, "wait for node ready")
		defer ktesting.End(tCtx)
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) []corev1.Node {
			nodes, err := tCtx.Client().CoreV1().Nodes().List(tCtx, metav1.ListOptions{})
			tCtx.ExpectNoError(err, "list nodes")
			return nodes.Items
		}).Should(gomega.ConsistOf(gomega.HaveField("Status.Conditions", gomega.ContainElement(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"Type":   gomega.Equal(corev1.NodeReady),
			"Status": gomega.Equal(corev1.ConditionTrue),
		})))))
	}
}

func (c *Cluster) checkHealthz(tCtx ktesting.TContext, cmd *Cmd, method, hostIP, port string) {
	url := fmt.Sprintf("%s://%s:%s/healthz", method, hostIP, port)
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) error {
		if !cmd.Running() {
			return gomega.StopTrying(fmt.Sprintf("%s stopped unexpectedly", cmd.Name))
		}
		// Like kube::util::wait_for_url in local-up-cluster.sh we use https,
		// but don't check the certificate.
		req, err := http.NewRequestWithContext(tCtx, http.MethodGet, url, nil)
		if err != nil {
			return fmt.Errorf("create request: %w", err)
		}
		tr := &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		}
		client := &http.Client{Transport: tr}
		resp, err := client.Do(req)
		if err != nil {
			return fmt.Errorf("get %s: %w", url, err)
		}
		if err := resp.Body.Close(); err != nil {
			return fmt.Errorf("close GET response: %w", err)
		}
		// Any response is fine, we just need to get here. In practice, we get a 403 Forbidden.
		return nil
	}).Should(gomega.Succeed(), fmt.Sprintf("HTTP GET %s", url))
}
