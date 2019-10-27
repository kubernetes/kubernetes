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

package kubectl

import (
	"bytes"
	"fmt"
	"io"
	"net"
	"net/url"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
	"time"

	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	uexec "k8s.io/utils/exec"
)

const defaultHost = "http://127.0.0.1:8080"

var (
	// For parsing Kubectl version for version-skewed testing.
	gitVersionRegexp = regexp.MustCompile("GitVersion:\"(v.+?)\"")
)

// KubectlBuilder is used to build, customize and execute a kubectl Command.
// Add more functions to customize the builder as needed.
type KubectlBuilder struct {
	// Cmd is the custom command executed by KubectlBuilder.
	Cmd     *exec.Cmd
	timeout <-chan time.Time
}

// TestKubeconfig is a struct containing the minimum attributes needed to run KubectlCmd.
type TestKubeconfig struct {
	CertDir     string
	Host        string
	KubeConfig  string
	KubeContext string
	KubectlPath string
}

// NewTestKubeconfig returns a new Kubeconfig struct instance.
func NewTestKubeconfig(certdir string, host string, kubeconfig string, kubecontext string, kubectlpath string) *TestKubeconfig {
	return &TestKubeconfig{
		CertDir:     certdir,
		Host:        host,
		KubeConfig:  kubeconfig,
		KubeContext: kubecontext,
		KubectlPath: kubectlpath,
	}
}

// KubectlCmd runs the kubectl executable through the wrapper script.
func KubectlCmd(certdir string, host string, kubeconfig string, kubecontext string, kubectlpath string, args ...string) *exec.Cmd {
	defaultArgs := []string{}

	// Reference a --server option so tests can run anywhere.
	if host != "" {
		defaultArgs = append(defaultArgs, "--"+clientcmd.FlagAPIServer+"="+host)
	}
	if kubeconfig != "" {
		defaultArgs = append(defaultArgs, "--"+clientcmd.RecommendedConfigPathFlag+"="+kubeconfig)

		// Reference the KubeContext
		if kubecontext != "" {
			defaultArgs = append(defaultArgs, "--"+clientcmd.FlagContext+"="+kubecontext)
		}

	} else {
		if certdir != "" {
			defaultArgs = append(defaultArgs,
				fmt.Sprintf("--certificate-authority=%s", filepath.Join(certdir, "ca.crt")),
				fmt.Sprintf("--client-certificate=%s", filepath.Join(certdir, "kubecfg.crt")),
				fmt.Sprintf("--client-key=%s", filepath.Join(certdir, "kubecfg.key")))
		}
	}
	kubectlArgs := append(defaultArgs, args...)

	//We allow users to specify path to kubectl, so you can test either "kubectl" or "cluster/kubectl.sh"
	//and so on.
	cmd := exec.Command(kubectlpath, kubectlArgs...)

	//caller will invoke this and wait on it.
	return cmd
}

// KubectlVersion gets the version of kubectl that's currently being used (see
// --kubectl-path in e2e.go to use an alternate kubectl).
func KubectlVersion(certdir string, host string, kubeconfig string, kubecontext string, kubectlpath string) (*utilversion.Version, error) {
	args := []string{"version", "--client"}
	output := RunKubectlOrDie(certdir, host, kubeconfig, kubecontext, kubectlpath, args...)
	matches := gitVersionRegexp.FindStringSubmatch(output)
	if len(matches) != 2 {
		return nil, fmt.Errorf("Could not find kubectl version in output %v", output)
	}
	// Don't use the full match, as it contains "GitVersion:\"" and a
	// trailing "\"".  Just use the submatch.
	return utilversion.ParseSemantic(matches[1])
}

// NewKubectlCommand returns a KubectlBuilder for running kubectl.
func NewKubectlCommand(certdir string, host string, kubeconfig string, kubecontext string, kubectlpath string, args ...string) *KubectlBuilder {
	b := new(KubectlBuilder)
	b.Cmd = KubectlCmd(certdir, host, kubeconfig, kubecontext, kubectlpath, args...)
	return b
}

// WithEnv sets the given environment and returns itself.
func (b *KubectlBuilder) WithEnv(env []string) *KubectlBuilder {
	b.Cmd.Env = env
	return b
}

// WithTimeout sets the given timeout and returns itself.
func (b *KubectlBuilder) WithTimeout(t <-chan time.Time) *KubectlBuilder {
	b.timeout = t
	return b
}

// WithStdinData sets the given data to stdin and returns itself.
func (b KubectlBuilder) WithStdinData(data string) *KubectlBuilder {
	b.Cmd.Stdin = strings.NewReader(data)
	return &b
}

// WithStdinReader sets the given reader and returns itself.
func (b KubectlBuilder) WithStdinReader(reader io.Reader) *KubectlBuilder {
	b.Cmd.Stdin = reader
	return &b
}

// Exec runs the kubectl executable.
func (b KubectlBuilder) Exec() (string, error) {
	var stdout, stderr bytes.Buffer
	cmd := b.Cmd
	cmd.Stdout, cmd.Stderr = &stdout, &stderr

	e2elog.Logf("Running '%s %s'", cmd.Path, strings.Join(cmd.Args[1:], " ")) // skip arg[0] as it is printed separately
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("error starting %v:\nCommand stdout:\n%v\nstderr:\n%v\nerror:\n%v", cmd, cmd.Stdout, cmd.Stderr, err)
	}
	errCh := make(chan error, 1)
	go func() {
		errCh <- cmd.Wait()
	}()
	select {
	case err := <-errCh:
		if err != nil {
			var rc = 127
			if ee, ok := err.(*exec.ExitError); ok {
				rc = int(ee.Sys().(syscall.WaitStatus).ExitStatus())
				e2elog.Logf("rc: %d", rc)
			}
			return "", uexec.CodeExitError{
				Err:  fmt.Errorf("error running %v:\nCommand stdout:\n%v\nstderr:\n%v\nerror:\n%v", cmd, cmd.Stdout, cmd.Stderr, err),
				Code: rc,
			}
		}
	case <-b.timeout:
		b.Cmd.Process.Kill()
		return "", fmt.Errorf("timed out waiting for command %v:\nCommand stdout:\n%v\nstderr:\n%v", cmd, cmd.Stdout, cmd.Stderr)
	}
	e2elog.Logf("stderr: %q", stderr.String())
	e2elog.Logf("stdout: %q", stdout.String())
	return stdout.String(), nil
}

// ExecOrDie runs the kubectl executable or dies if error occurs.
func (b KubectlBuilder) ExecOrDie(certdir string, host string, kubeconfig string, kubecontext string, kubectlpath string) string {
	str, err := b.Exec()
	// In case of i/o timeout error, try talking to the apiserver again after 2s before dying.
	// Note that we're still dying after retrying so that we can get visibility to triage it further.
	if isTimeout(err) {
		e2elog.Logf("Hit i/o timeout error, talking to the server 2s later to see if it's temporary.")
		time.Sleep(2 * time.Second)
		tk := NewTestKubeconfig(certdir, host, kubeconfig, kubecontext, kubectlpath)
		retryStr, retryErr := RunKubectl(tk.CertDir, tk.Host, tk.KubeConfig, tk.KubeContext, tk.KubectlPath, "version")
		e2elog.Logf("stdout: %q", retryStr)
		e2elog.Logf("err: %v", retryErr)
	}
	expectNoError(err)
	return str
}

// TODO: move to its own sub-package
// expectNoError checks if "err" is set, and if so, fails assertion while logging the error.
func expectNoError(err error, explain ...interface{}) {
	expectNoErrorWithOffset(1, err, explain...)
}

// TODO: move to its own sub-package
// expectNoErrorWithOffset checks if "err" is set, and if so, fails assertion while logging the error at "offset" levels above its caller
// (for example, for call chain f -> g -> ExpectNoErrorWithOffset(1, ...) error would be logged for "f").
func expectNoErrorWithOffset(offset int, err error, explain ...interface{}) {
	if err != nil {
		e2elog.Logf("Unexpected error occurred: %v", err)
	}
	gomega.ExpectWithOffset(1+offset, err).NotTo(gomega.HaveOccurred(), explain...)
}

// TODO: move to its own sub-package
// expectNoErrorWithRetries checks if an error occurs with the given retry count.
func expectNoErrorWithRetries(fn func() error, maxRetries int, explain ...interface{}) {
	var err error
	for i := 0; i < maxRetries; i++ {
		err = fn()
		if err == nil {
			return
		}
		e2elog.Logf("(Attempt %d of %d) Unexpected error occurred: %v", i+1, maxRetries, err)
	}
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred(), explain...)
}

func isTimeout(err error) bool {
	switch err := err.(type) {
	case net.Error:
		if err.Timeout() {
			return true
		}
	case *url.Error:
		if err, ok := err.Err.(net.Error); ok && err.Timeout() {
			return true
		}
	}
	return false
}

func KubectlLogPod(c clientset.Interface, pod v1.Pod, containerNameSubstr string, logFunc func(ftm string, args ...interface{})) {
	for _, container := range pod.Spec.Containers {
		if strings.Contains(container.Name, containerNameSubstr) {
			// Contains() matches all strings if substr is empty
			logs, err := e2epod.GetPodLogs(c, pod.Namespace, pod.Name, container.Name)
			if err != nil {
				logs, err = e2epod.GetPreviousPodLogs(c, pod.Namespace, pod.Name, container.Name)
				if err != nil {
					logFunc("Failed to get logs of pod %v, container %v, err: %v", pod.Name, container.Name, err)
				}
			}
			logFunc("Logs of %v/%v:%v on node %v", pod.Namespace, pod.Name, container.Name, pod.Spec.NodeName)
			logFunc("%s : STARTLOG\n%s\nENDLOG for container %v:%v:%v", containerNameSubstr, logs, pod.Namespace, pod.Name, container.Name)
		}
	}
}

// RunKubectlOrDie is a convenience wrapper over KubectlBuilder
func RunKubectlOrDie(certdir string, host string, kubeconfig string, kubecontext string, kubectlpath string, args ...string) string {
	return NewKubectlCommand(certdir, host, kubeconfig, kubecontext, kubectlpath, args...).ExecOrDie(certdir, host, kubeconfig, kubecontext, kubectlpath)
}

// RunKubectl is a convenience wrapper over KubectlBuilder
func RunKubectl(certdir string, host string, kubeconfig string, kubecontext string, kubectlpath string, args ...string) (string, error) {
	return NewKubectlCommand(certdir, host, kubeconfig, kubecontext, kubectlpath, args...).Exec()
}

// RunKubectlOrDieInput is a convenience wrapper over KubectlBuilder that takes input to stdin
func RunKubectlOrDieInput(certdir string, host string, kubeconfig string, kubecontext string, kubectlpath string, data string, args ...string) string {
	return NewKubectlCommand(certdir, host, kubeconfig, kubecontext, kubectlpath, args...).WithStdinData(data).ExecOrDie(certdir, host, kubeconfig, kubecontext, kubectlpath)
}

// RunKubectlInput is a convenience wrapper over KubectlBuilder that takes input to stdin
func RunKubectlInput(certdir string, host string, kubeconfig string, kubecontext string, kubectlpath string, data string, args ...string) (string, error) {
	return NewKubectlCommand(certdir, host, kubeconfig, kubecontext, kubectlpath, args...).WithStdinData(data).Exec()
}
