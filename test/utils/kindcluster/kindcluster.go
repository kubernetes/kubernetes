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

// Package kindcluster contains wrapper code around invoking kind
// and managing the resulting cluster.
//
// E2E tests using this must be labeled with feature.KindCommand.
package kindcluster

import (
	"bufio"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/uuid"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	gtypes "github.com/onsi/gomega/types"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	appsv1ac "k8s.io/client-go/applyconfigurations/apps/v1"
	corev1ac "k8s.io/client-go/applyconfigurations/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/utils/format"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func New(tCtx ktesting.TContext) *Cluster {
	c := &Cluster{}
	tCtx.CleanupCtx(c.Stop)
	return c
}

// Cluster represents one kind cluster. Each instance is
// independent from all other instances as long the name
// chosen by the caller is unique on the node.
type Cluster struct {
	running    bool
	name       string
	dir        string
	kubeConfig string
}

// Start brings up the cluster anew. If it was already running, it will be stopped first.

// Will be stopped automatically at the end of the test.
// If the ARTIFACTS env variable is set and the test failed,
// log files of the kind cluster get dumped into
// $ARTIFACTS/<test name>/kind/<cluster name> before stopping it.
//
// The name should be unique to enable running clusters in parallel.
// The config has to be a complete YAML according to https://kind.sigs.k8s.io/docs/user/configuration/.
// The image source can be a directory containing Kubernetes source code or
// a download URL like https://dl.k8s.io/ci/v1.33.1-19+f900f017250646/kubernetes-server-linux-amd64.tar.gz.
func (c *Cluster) Start(tCtx ktesting.TContext, name, kindConfig, imageName string) {
	tCtx.Helper()
	if c.running {
		c.Stop(tCtx)
	}
	c.name = name
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		// Intentional additional lambda function for source code location in log output.
		c.Stop(tCtx)
	})
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		artifacts := os.ExpandEnv("${ARTIFACTS}")
		if c.name == "" || artifacts == "" || !tCtx.Failed() {
			return
		}
		// Sanitize the name:
		// - remove E2E [] tags
		// - replace whitespaces and some special characters with hyphens
		testName := tCtx.Name()
		testName = regexp.MustCompile(`\s*\[[^\]]*\]`).ReplaceAllString(testName, "")
		testName = regexp.MustCompile(`[[:space:]/:()\\]+`).ReplaceAllString(testName, "-")
		testName = strings.Trim(testName, "-")
		logDir := path.Join(artifacts, testName, "kind", c.name)
		tCtx.Logf("exporting kind cluster logs for %s to %s after test failure", c.name, logDir)
		runKind(tCtx, "export", "logs", "--name", c.name, logDir)
	})

	c.dir = tCtx.TempDir()
	config := path.Join(c.dir, "kind.yaml")
	tCtx.ExpectNoError(os.WriteFile(config, []byte(kindConfig), 0644), "create kind config file")
	c.kubeConfig = path.Join(c.dir, "kube.config")
	// --retain ensures that log exporting above works in case of a failure.
	runKind(tCtx, "create", "cluster", "--retain", "--name", name, "--config", config, "--kubeconfig", c.kubeConfig, "--image", imageName)
	c.running = true
	tCtx.Logf("cluster is running, use KUBECONFIG=%s to access it", c.kubeConfig)
}

// Stop ensures that the cluster is not running anymore.
func (c *Cluster) Stop(tCtx ktesting.TContext) {
	tCtx.Helper()
	runKind(tCtx, "delete", "cluster", "--name", c.name)
	c.running = false
	c.name = ""
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

type ModifyOptions struct {
	// DockerTag changes the image version used by all components.
	// Left unchanged if empty.
	DockerTag string

	// DockerTagByComponent overrides DockerTag for specific components
	// (kube-apiserver, kube-controller-manager, kube-scheduler, kube-proxy).
	DockerTagByComponent map[string]string

	// ArchSuffix (for example, "-amd64") gets appended to image names.
	// Empty for multi-platform images.
	ArchSuffix string

	// ArchSuffixByComponent overrides ArchSuffix for specific components.
	ArchSuffixByComponent map[string]string

	// ReleaseImagesDir specifies where to find the image .tar files.
	// If set, those images will be side-loaded into nodes as needed,
	// otherwise the nodes must have it or be able to pull it.
	ReleaseImagesDir string

	// KubeletBinary is the path to a kubelet binary which will be
	// used to update worker nodes. If unset, the kubelet does not
	// get updated.
	//
	// If this starts with a colon, the file is stored under the
	// path without the colon on each worker node.
	//
	// The control plane node is not getting updated to speed up
	// the process. The version of the kubelet there shouldn't matter.
	KubeletBinary string

	// Upgrade determines whether the apiserver gets updated first (upgrade)
	// or last (downgrade).
	Upgrade bool
}

func (m ModifyOptions) GetDockerTag(component string) string {
	if tag, ok := m.DockerTagByComponent[component]; ok {
		return tag
	}
	return m.DockerTag
}

func (m ModifyOptions) GetArchSuffix(component string) string {
	if tag, ok := m.ArchSuffixByComponent[component]; ok {
		return tag
	}
	return m.ArchSuffix
}

// GetSystemLogs returns the log output of a control plane component.
func (c *Cluster) GetSystemLogs(tCtx ktesting.TContext, component string) string {
	tCtx.Helper()

	namespace := "kube-system"
	podName := component + "-" + c.name + "-control-plane"
	pod, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, podName, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get pod %s/%s", namespace, podName)

	request := tCtx.Client().CoreV1().RESTClient().Get().
		Resource("pods").
		Namespace(namespace).
		Name(podName).SubResource("log").
		Param("container", pod.Spec.Containers[0].Name)
	logs, err := request.Do(tCtx).Raw()
	tCtx.ExpectNoError(err, "get pod %s/%s container %s logs", namespace, podName, pod.Spec.Containers[0].Name)
	return string(logs)
}

// Modify changes the cluster as described in the options.
// It returns options that can be passed to Modify unchanged
// to restore the original state.
//
// This uses the same approach as https://gist.github.com/aojea/2c94034f8e86d08842e5916231eb3fe1.
func (c *Cluster) Modify(tCtx ktesting.TContext, options ModifyOptions) ModifyOptions {
	tCtx.Helper()

	restore := ModifyOptions{
		DockerTagByComponent:  make(map[string]string),
		ArchSuffixByComponent: make(map[string]string),
	}

	// Ensure that we have a client for the cluster.
	restConfig := c.LoadConfig(tCtx)
	restConfig.UserAgent = fmt.Sprintf("%s -- kindcluster", restclient.DefaultKubernetesUserAgent())
	tCtx = ktesting.WithRESTConfig(tCtx, restConfig)

	// Determine nodes.
	nodes := regexp.MustCompile(`[[:space:]]+`).Split(strings.TrimSpace(runKind(tCtx, "get", "nodes", "--name", c.name)), -1)
	workerNodes := make([]string, 0, len(nodes)-1)
	controlPlaneNodes := make([]string, 0, 1)
	for _, node := range nodes {
		if strings.Contains(node, "worker") {
			workerNodes = append(workerNodes, node)
		} else {
			controlPlaneNodes = append(controlPlaneNodes, node)
		}
	}

	restore.Upgrade = !options.Upgrade
	if options.Upgrade {
		c.modifyControlPlane(tCtx, options, controlPlaneNodes, []string{"kube-apiserver", "kube-controller-manager", "kube-scheduler"}, &restore)
		c.modifyKubelets(tCtx, options, workerNodes, &restore)
		c.modifyKubeProxy(tCtx, options, nodes, &restore)
	} else {
		c.modifyKubeProxy(tCtx, options, nodes, &restore)
		c.modifyKubelets(tCtx, options, workerNodes, &restore)
		c.modifyControlPlane(tCtx, options, controlPlaneNodes, []string{"kube-scheduler", "kube-controller-manager", "kube-apiserver"}, &restore)
	}

	return restore
}

func (c *Cluster) modifyControlPlane(tCtx ktesting.TContext, options ModifyOptions, controlPlaneNodes, controlPlaneComponents []string, restore *ModifyOptions) {
	tCtx.Helper()

	for _, node := range controlPlaneNodes {
		for _, component := range controlPlaneComponents {
			if releaseImagesDir := options.ReleaseImagesDir; releaseImagesDir != "" {
				runKind(tCtx, "load", "image-archive", path.Join(releaseImagesDir, component+".tar"), "--name", c.name, "--nodes", node)
			}
			if dockerTag := options.GetDockerTag(component); dockerTag != "" {
				manifestPath := "/etc/kubernetes/manifests/" + component + ".yaml"
				manifest := runCmd(tCtx, "docker", "exec", node, "cat", manifestPath)
				match := regexp.MustCompile(`(?m)^([[:space:]]*image:.+?)(-amd64|-arm64|-arm|-ppc64le|-s390x)?:(.+)$`).FindStringSubmatchIndex(manifest)
				if match == nil {
					tCtx.Fatalf("image specification not found in %s manifest on node %s:\n%s", component, node, manifest)
				}
				modifiedManifest := manifest[:match[0]] +
					fmt.Sprintf("%s%s:%s", manifest[match[2]:match[3]], options.GetArchSuffix(component), dockerTag) +
					manifest[match[1]:]
				// This is not tracked separately by node, which shouldn't be necessary.
				oldArchSuffix := ""
				if match[4] != -1 {
					oldArchSuffix = manifest[match[4]:match[5]]
				}
				restore.ArchSuffixByComponent[component] = oldArchSuffix
				restore.DockerTagByComponent[component] = manifest[match[6]:match[7]]
				tCtx.Logf("Patching %s on node %s. Before:\n%s\n\nDiff (- old, + new):\n%s", manifestPath, node, manifest, cmp.Diff(manifest, modifiedManifest))
				runCmdWithInput(tCtx, strings.NewReader(modifiedManifest), "docker", "exec", "-i", node, "dd", "of="+manifestPath)
				ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *corev1.Pod {
					pod, err := tCtx.Client().CoreV1().Pods("kube-system").Get(tCtx, fmt.Sprintf("%s-%s-control-plane", component, c.name), metav1.GetOptions{})
					tCtx.ExpectNoError(err)
					return pod
				}).WithTimeout(5*time.Minute).Should(useImage(options.DockerTag), "%s should have restarted with image %s", component, dockerTag)
			}
		}
	}
}

func (c *Cluster) modifyKubelets(tCtx ktesting.TContext, options ModifyOptions, workerNodes []string, restore *ModifyOptions) {
	tCtx.Helper()

	if kubeletBinary := options.KubeletBinary; kubeletBinary != "" {
		// To support restore, each kubelet modification makes a new
		// copy of the original state under a unique name which is the
		// same for all nodes. We could try to hash the content to
		// avoid accumulating cruft, but that doesn't seem worth it.
		uuid := uuid.New().String()
		restore.KubeletBinary = ":/usr/bin/kubelet." + uuid
		for _, node := range workerNodes {
			runAndLogCmd(tCtx, "docker", "exec", node, "cp", "-a", "/usr/bin/kubelet", restore.KubeletBinary[1:])
			if strings.HasPrefix(kubeletBinary, ":") {
				// Text file busy, need to remove it first.
				runAndLogCmd(tCtx, "docker", "exec", node, "cp", "-af", kubeletBinary[1:], "/usr/bin/kubelet")
			} else {
				runAndLogCmd(tCtx, "docker", "cp", kubeletBinary, node+":/usr/bin/kubelet")
			}
			// systemctl restart checks that the service comes up.
			runAndLogCmd(tCtx, "docker", "exec", node, "systemctl", "restart", "kubelet")
		}
	}
}

func (c *Cluster) modifyKubeProxy(tCtx ktesting.TContext, options ModifyOptions, nodes []string, restore *ModifyOptions) {
	tCtx.Helper()

	if releaseImagesDir := options.ReleaseImagesDir; releaseImagesDir != "" {
		runKind(tCtx, "load", "image-archive", path.Join(releaseImagesDir, "kube-proxy.tar"), "--name", c.name, "--nodes", strings.Join(nodes, ","))
	}
	if dockerTag := options.GetDockerTag("kube-proxy"); dockerTag != "" {
		oldImage, kubeProxyDS := setDaemonSetImage(tCtx, types.NamespacedName{Namespace: "kube-system", Name: "kube-proxy"}, "kube-proxy", fmt.Sprintf("registry.k8s.io/kube-proxy%s:%s", options.ArchSuffix, dockerTag))
		parts := regexp.MustCompile(`(.+?)(-amd64|-arm64|-arm|-ppc64le|-s390x)?:(.+)`).FindStringSubmatch(oldImage)
		if parts == nil {
			tCtx.Fatalf("could not split image specification: %q", oldImage)
		}
		restore.ArchSuffixByComponent["kube-proxy"] = parts[2]
		restore.DockerTagByComponent["kube-proxy"] = parts[3]
		waitForDSRollout(tCtx, kubeProxyDS, len(nodes))
	}
}

// useImage asserts that a pod's container runs with a certain image, identified via a substring (for example, the image version),
// and is ready.
func useImage(imageSubString string) gtypes.GomegaMatcher {
	return gomega.HaveField("Status.ContainerStatuses", gomega.ConsistOf(
		gomega.And(
			gomega.HaveField("State.Running", gomega.Not(gomega.BeNil())),
			gomega.HaveField("Image", gomega.ContainSubstring(imageSubString)),
			gomega.HaveField("Ready", gomega.Equal(true)),
		),
	))
}

// setDaemonSetImage implements "kubectl set image daemonset".
func setDaemonSetImage(tCtx ktesting.TContext, ds types.NamespacedName, containerName, image string) (string, *appsv1.DaemonSet) {
	tCtx.Helper()
	oldDS, err := tCtx.Client().AppsV1().DaemonSets(ds.Namespace).Get(tCtx, ds.Name, metav1.GetOptions{})
	oldContainer := slices.IndexFunc(oldDS.Spec.Template.Spec.Containers, func(container corev1.Container) bool {
		return container.Name == containerName
	})
	if oldContainer < 0 {
		tCtx.Fatalf("container %s not found in DaemonSet:\n%s", format.Object(oldDS, 1))
	}
	oldImage := oldDS.Spec.Template.Spec.Containers[oldContainer].Image
	tCtx.ExpectNoError(err, "get DaemonSet")
	dsAC := appsv1ac.DaemonSet(ds.Name, ds.Namespace).WithSpec(
		appsv1ac.DaemonSetSpec().WithTemplate(
			corev1ac.PodTemplateSpec().WithSpec(
				corev1ac.PodSpec().WithContainers(
					corev1ac.Container().WithName(containerName).WithImage(image),
				),
			),
		),
	)
	patchedDS, err := tCtx.Client().AppsV1().DaemonSets(ds.Namespace).Apply(tCtx, dsAC, metav1.ApplyOptions{FieldManager: "kindcluster", Force: true})
	tCtx.ExpectNoError(err, "update image in DaemonSet %s", ds)
	tCtx.Logf("updated:\n%s", format.Object(patchedDS, 1))
	return oldImage, patchedDS
}

// waitForDSRollout checks that the DaemonSet controller has started all pods.
func waitForDSRollout(tCtx ktesting.TContext, ds *appsv1.DaemonSet, numNodes int) {
	tCtx.Helper()

	tCtx = ktesting.Begin(tCtx, fmt.Sprintf("%s with Generation %d", klog.KObj(ds), ds.Generation))
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *appsv1.DaemonSet {
		ds, err := tCtx.Client().AppsV1().DaemonSets(ds.Namespace).Get(tCtx, ds.Name, metav1.GetOptions{})
		tCtx.ExpectNoError(err)
		return ds
	}).WithTimeout(time.Duration(numNodes) * 3 * time.Minute).Should(gomega.HaveField("Status", gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
		"CurrentNumberScheduled": gomega.Equal(int32(numNodes)),
		"NumberMisscheduled":     gomega.Equal(int32(0)),
		"DesiredNumberScheduled": gomega.Equal(int32(numNodes)),
		"NumberReady":            gomega.Equal(int32(numNodes)),
		"UpdatedNumberScheduled": gomega.Equal(int32(numNodes)),
		"ObservedGeneration":     gomega.BeNumerically(">=", ds.Generation),
	})))
	ktesting.End(tCtx)
}

// BuildImage ensures that there is a kind node image for the given version
// of Kubernetes. It can build from a directory with Kubernetes source code
// or download URL like https://dl.k8s.io/ci/v1.33.1-19+f900f017250646/kubernetes-server-linux-amd64.tar.gz.
// It returns the image name required to start a cluster.
func BuildImage(tCtx ktesting.TContext, version, imageSource string) string {
	tCtx.Helper()
	imageName := "kind/cluster:" + version
	// Comment this out temporarily when developing locally to save some time
	// if the image was already built before.
	runKind(tCtx, "build", "node-image", "--image", imageName, imageSource)
	return imageName
}

// ServerDownloadURL returns the full URL from which kind can create a node image
// for the given major/minor version of Kubernetes.
//
// This considers only proper releases.
func ServerDownloadURL(tCtx ktesting.TContext, major, minor uint) string {
	tCtx.Helper()
	url := fmt.Sprintf("https://dl.k8s.io/release/stable-%d.%d.txt", major, minor)
	get, err := http.NewRequestWithContext(tCtx, http.MethodGet, url, nil)
	tCtx.ExpectNoError(err, "construct GET for %s", url)
	resp, err := http.DefaultClient.Do(get)
	tCtx.ExpectNoError(err, "get %s", url)
	if resp.StatusCode != http.StatusOK {
		tCtx.Fatalf("get %s: %d - %s", url, resp.StatusCode, resp.Status)
	}
	if resp.Body == nil {
		tCtx.Fatalf("empty response for %s", url)
	}
	defer func() {
		tCtx.ExpectNoError(resp.Body.Close(), "close response body")
	}()
	version, err := io.ReadAll(resp.Body)
	tCtx.ExpectNoError(err, "read response body for %s", url)
	return fmt.Sprintf("https://dl.k8s.io/release/%s/kubernetes-server-linux-amd64.tar.gz", string(version))
}

func runKind(tCtx ktesting.TContext, args ...string) string {
	tCtx.Helper()
	return runAndLogCmd(tCtx, "kind", args...)
}

func runAndLogCmd(tCtx ktesting.TContext, name string, args ...string) string {
	tCtx.Helper()
	tCtx.Logf("Running command: %s %s", name, strings.Join(args, " "))
	cmd := exec.CommandContext(tCtx, name, args...)
	var output strings.Builder
	reader, writer := io.Pipe()
	cmd.Stdout = writer
	cmd.Stderr = writer
	tCtx.ExpectNoError(cmd.Start(), "start %s command", name)
	scanner := bufio.NewScanner(reader)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for scanner.Scan() {
			line := scanner.Text()
			line = strings.TrimSuffix(line, "\n")
			tCtx.Logf("%s: %s", name, line)
			output.WriteString(line)
			output.WriteByte('\n')
		}
	}()
	tCtx.ExpectNoError(cmd.Wait(), fmt.Sprintf("%s command failed, output:\n%s", name, output.String()))
	tCtx.ExpectNoError(writer.Close(), "close in-memory pipe")
	wg.Wait()
	tCtx.ExpectNoError(scanner.Err(), "read %s command output", name)

	return output.String()
}

func runCmd(tCtx ktesting.TContext, name string, args ...string) string {
	tCtx.Helper()
	tCtx.Logf("Running command: %s %s", name, strings.Join(args, " "))
	cmd := exec.CommandContext(tCtx, name, args...)
	output, err := cmd.CombinedOutput()
	tCtx.ExpectNoError(err, "command %s failed\noutput:\n%s", name, string(output))
	return string(output)
}

func runCmdWithInput(tCtx ktesting.TContext, input io.Reader, name string, args ...string) string {
	tCtx.Helper()
	tCtx.Logf("Running command: %s %s", name, strings.Join(args, " "))
	cmd := exec.CommandContext(tCtx, name, args...)
	cmd.Stdin = input
	output, err := cmd.CombinedOutput()
	tCtx.ExpectNoError(err, "command %s failed\noutput:\n%s", name, string(output))
	return string(output)
}
