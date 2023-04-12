/*
Copyright 2019 The Kubernetes Authors.

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
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"

	"github.com/onsi/ginkgo/v2"
)

const (
	maxKubectlExecRetries = 5
)

// TestKubeconfig is a struct containing the needed attributes from TestContext and Framework(Namespace).
type TestKubeconfig struct {
	CertDir     string
	Host        string
	KubeConfig  string
	KubeContext string
	KubectlPath string
	Namespace   string // Every test has at least one namespace unless creation is skipped
}

// NewTestKubeconfig returns a new Kubeconfig struct instance.
func NewTestKubeconfig(certdir, host, kubeconfig, kubecontext, kubectlpath, namespace string) *TestKubeconfig {
	return &TestKubeconfig{
		CertDir:     certdir,
		Host:        host,
		KubeConfig:  kubeconfig,
		KubeContext: kubecontext,
		KubectlPath: kubectlpath,
		Namespace:   namespace,
	}
}

// KubectlCmd runs the kubectl executable through the wrapper script.
func (tk *TestKubeconfig) KubectlCmd(args ...string) *exec.Cmd {
	defaultArgs := []string{}

	// Reference a --server option so tests can run anywhere.
	if tk.Host != "" {
		defaultArgs = append(defaultArgs, "--"+clientcmd.FlagAPIServer+"="+tk.Host)
	}
	if tk.KubeConfig != "" {
		defaultArgs = append(defaultArgs, "--"+clientcmd.RecommendedConfigPathFlag+"="+tk.KubeConfig)

		// Reference the KubeContext
		if tk.KubeContext != "" {
			defaultArgs = append(defaultArgs, "--"+clientcmd.FlagContext+"="+tk.KubeContext)
		}

	} else {
		if tk.CertDir != "" {
			defaultArgs = append(defaultArgs,
				fmt.Sprintf("--certificate-authority=%s", filepath.Join(tk.CertDir, "ca.crt")),
				fmt.Sprintf("--client-certificate=%s", filepath.Join(tk.CertDir, "kubecfg.crt")),
				fmt.Sprintf("--client-key=%s", filepath.Join(tk.CertDir, "kubecfg.key")))
		}
	}
	if tk.Namespace != "" {
		defaultArgs = append(defaultArgs, fmt.Sprintf("--namespace=%s", tk.Namespace))
	}
	kubectlArgs := append(defaultArgs, args...)

	//We allow users to specify path to kubectl, so you can test either "kubectl" or "cluster/kubectl.sh"
	//and so on.
	cmd := exec.Command(tk.KubectlPath, kubectlArgs...)

	//caller will invoke this and wait on it.
	return cmd
}

// LogFailedContainers runs `kubectl logs` on a failed containers.
func LogFailedContainers(ctx context.Context, c clientset.Interface, ns string, logFunc func(ftm string, args ...interface{})) {
	podList, err := c.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{})
	if err != nil {
		logFunc("Error getting pods in namespace '%s': %v", ns, err)
		return
	}
	logFunc("Running kubectl logs on non-ready containers in %v", ns)
	for _, pod := range podList.Items {
		if res, err := testutils.PodRunningReady(&pod); !res || err != nil {
			kubectlLogPod(ctx, c, pod, "", framework.Logf)
		}
	}
}

func kubectlLogPod(ctx context.Context, c clientset.Interface, pod v1.Pod, containerNameSubstr string, logFunc func(ftm string, args ...interface{})) {
	for _, container := range pod.Spec.Containers {
		if strings.Contains(container.Name, containerNameSubstr) {
			// Contains() matches all strings if substr is empty
			logs, err := e2epod.GetPodLogs(ctx, c, pod.Namespace, pod.Name, container.Name)
			if err != nil {
				logs, err = e2epod.GetPreviousPodLogs(ctx, c, pod.Namespace, pod.Name, container.Name)
				if err != nil {
					logFunc("Failed to get logs of pod %v, container %v, err: %v", pod.Name, container.Name, err)
				}
			}
			logFunc("Logs of %v/%v:%v on node %v", pod.Namespace, pod.Name, container.Name, pod.Spec.NodeName)
			logFunc("%s : STARTLOG\n%s\nENDLOG for container %v:%v:%v", containerNameSubstr, logs, pod.Namespace, pod.Name, container.Name)
		}
	}
}

// WriteFileViaContainer writes a file using kubectl exec echo <contents> > <path> via specified container
// because of the primitive technique we're using here, we only allow ASCII alphanumeric characters
func (tk *TestKubeconfig) WriteFileViaContainer(podName, containerName string, path string, contents string) error {
	ginkgo.By("writing a file in the container")
	allowedCharacters := "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
	for _, c := range contents {
		if !strings.ContainsRune(allowedCharacters, c) {
			return fmt.Errorf("Unsupported character in string to write: %v", c)
		}
	}
	command := fmt.Sprintf("echo '%s' > '%s'; sync", contents, path)
	stdout, stderr, err := tk.kubectlExecWithRetry(tk.Namespace, podName, containerName, "--", "/bin/sh", "-c", command)
	if err != nil {
		framework.Logf("error running kubectl exec to write file: %v\nstdout=%v\nstderr=%v)", err, string(stdout), string(stderr))
	}
	return err
}

// ReadFileViaContainer reads a file using kubectl exec cat <path>.
func (tk *TestKubeconfig) ReadFileViaContainer(podName, containerName string, path string) (string, error) {
	ginkgo.By("reading a file in the container")

	stdout, stderr, err := tk.kubectlExecWithRetry(tk.Namespace, podName, containerName, "--", "cat", path)
	if err != nil {
		framework.Logf("error running kubectl exec to read file: %v\nstdout=%v\nstderr=%v)", err, string(stdout), string(stderr))
	}
	return string(stdout), err
}

func (tk *TestKubeconfig) kubectlExecWithRetry(namespace string, podName, containerName string, args ...string) ([]byte, []byte, error) {
	for numRetries := 0; numRetries < maxKubectlExecRetries; numRetries++ {
		if numRetries > 0 {
			framework.Logf("Retrying kubectl exec (retry count=%v/%v)", numRetries+1, maxKubectlExecRetries)
		}

		stdOutBytes, stdErrBytes, err := tk.kubectlExec(namespace, podName, containerName, args...)
		if err != nil {
			if strings.Contains(strings.ToLower(string(stdErrBytes)), "i/o timeout") {
				// Retry on "i/o timeout" errors
				framework.Logf("Warning: kubectl exec encountered i/o timeout.\nerr=%v\nstdout=%v\nstderr=%v)", err, string(stdOutBytes), string(stdErrBytes))
				continue
			}
			if strings.Contains(strings.ToLower(string(stdErrBytes)), "container not found") {
				// Retry on "container not found" errors
				framework.Logf("Warning: kubectl exec encountered container not found.\nerr=%v\nstdout=%v\nstderr=%v)", err, string(stdOutBytes), string(stdErrBytes))
				time.Sleep(2 * time.Second)
				continue
			}
		}

		return stdOutBytes, stdErrBytes, err
	}
	err := fmt.Errorf("Failed: kubectl exec failed %d times with \"i/o timeout\". Giving up", maxKubectlExecRetries)
	return nil, nil, err
}

func (tk *TestKubeconfig) kubectlExec(namespace string, podName, containerName string, args ...string) ([]byte, []byte, error) {
	var stdout, stderr bytes.Buffer
	cmdArgs := []string{
		"exec",
		fmt.Sprintf("--namespace=%v", namespace),
		podName,
		fmt.Sprintf("-c=%v", containerName),
	}
	cmdArgs = append(cmdArgs, args...)

	cmd := tk.KubectlCmd(cmdArgs...)
	cmd.Stdout, cmd.Stderr = &stdout, &stderr

	framework.Logf("Running '%s %s'", cmd.Path, strings.Join(cmdArgs, " "))
	err := cmd.Run()
	return stdout.Bytes(), stderr.Bytes(), err
}
