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

package apiclient

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"text/template"
	"time"

	"github.com/lithammer/dedent"
	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	netutil "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

const (
	// TODO: switch to /livez once all components support it
	// and delete the endpointHealthz constant.
	// https://github.com/kubernetes/kubernetes/issues/118158
	endpointHealthz = "healthz"
	endpointLivez   = "livez"

	argPort        = "secure-port"
	argBindAddress = "bind-address"
	// By default, for kube-api-server, kubeadm does not apply a --bind-address flag.
	// Check --advertise-address instead.
	argAdvertiseAddress = "advertise-address"
)

var (
	controlPlaneFailTempl = template.Must(template.New("init").Parse(dedent.Dedent(`
	A control plane component may have crashed or exited when started by the container runtime.
	To troubleshoot, list all containers using your preferred container runtimes CLI.
	Here is one example how you may list all running Kubernetes containers by using crictl:
		- 'crictl --runtime-endpoint {{ .Socket }} ps -a | grep kube | grep -v pause'
		Once you have found the failing container, you can inspect its logs with:
		- 'crictl --runtime-endpoint {{ .Socket }} logs CONTAINERID'
`)))

	kubeletFailMsg = dedent.Dedent(`
	Unfortunately, an error has occurred, likely caused by:
		- The kubelet is not running
		- The kubelet is unhealthy due to a misconfiguration of the node in some way (required cgroups disabled)

	If you are on a systemd-powered system, you can try to troubleshoot the error with the following commands:
		- 'systemctl status kubelet'
		- 'journalctl -xeu kubelet'
`)
)

// Waiter is an interface for waiting for criteria in Kubernetes to happen
type Waiter interface {
	// WaitForControlPlaneComponents waits for all control plane components to be ready.
	WaitForControlPlaneComponents(podMap map[string]*v1.Pod, apiServerAddress string) error
	// WaitForAPI waits for the API Server's /healthz endpoint to become "ok"
	// TODO: remove WaitForAPI once WaitForAllControlPlaneComponents goes GA:
	// https://github.com/kubernetes/kubeadm/issues/2907
	WaitForAPI() error
	// WaitForPodsWithLabel waits for Pods in the kube-system namespace to become Ready
	WaitForPodsWithLabel(kvLabel string) error
	// WaitForStaticPodSingleHash fetches sha256 hash for the control plane static pod
	WaitForStaticPodSingleHash(nodeName string, component string) (string, error)
	// WaitForStaticPodHashChange waits for the given static pod component's static pod hash to get updated.
	// By doing that we can be sure that the kubelet has restarted the given Static Pod
	WaitForStaticPodHashChange(nodeName, component, previousHash string) error
	// WaitForStaticPodControlPlaneHashes fetches sha256 hashes for the control plane static pods
	WaitForStaticPodControlPlaneHashes(nodeName string) (map[string]string, error)
	// WaitForKubelet blocks until the kubelet /healthz endpoint returns 'ok'
	WaitForKubelet(healthzAddress string, healthzPort int32) error
	// SetTimeout adjusts the timeout to the specified duration
	SetTimeout(timeout time.Duration)
}

// KubeWaiter is an implementation of Waiter that is backed by a Kubernetes client
type KubeWaiter struct {
	client  clientset.Interface
	timeout time.Duration
	writer  io.Writer
}

// NewKubeWaiter returns a new Waiter object that talks to the given Kubernetes cluster
func NewKubeWaiter(client clientset.Interface, timeout time.Duration, writer io.Writer) Waiter {
	return &KubeWaiter{
		client:  client,
		timeout: timeout,
		writer:  writer,
	}
}

// controlPlaneComponent holds a component name and an URL
// on which to perform health checks.
type controlPlaneComponent struct {
	name string
	url  string
}

// getControlPlaneComponentAddressAndPort parses the command in a static Pod
// container and extracts the values of the given args.
func getControlPlaneComponentAddressAndPort(pod *v1.Pod, name string, args []string) ([]string, error) {
	var (
		values    = make([]string, len(args))
		container *v1.Container
	)

	if pod == nil {
		return values, errors.Errorf("got nil Pod for component %q", name)
	}

	for i, c := range pod.Spec.Containers {
		if len(c.Command) == 0 {
			continue
		}
		if c.Command[0] == name {
			container = &pod.Spec.Containers[i]
			break
		}
	}
	if container == nil {
		return values, errors.Errorf("the Pod has no container command starting with %q", name)
	}

	for _, line := range container.Command {
		for i, arg := range args {
			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "--"+arg) && !strings.HasPrefix(line, "-"+arg) {
				continue
			}
			_, value, found := strings.Cut(line, "=")
			if !found {
				_, value, _ = strings.Cut(line, " ")
			}
			values[i] = value
		}
	}
	return values, nil
}

// getControlPlaneComponents reads the static Pods of control plane components
// and returns a slice of 'controlPlaneComponent'.
func getControlPlaneComponents(podMap map[string]*v1.Pod, addressAPIServer string) ([]controlPlaneComponent, error) {
	var (
		// By default kubeadm deploys the kube-controller-manager and kube-scheduler
		// with --bind-address=127.0.0.1. This should match get{Scheduler|ControllerManager}Command().
		addressKCM       = "127.0.0.1"
		addressScheduler = "127.0.0.1"

		portAPIServer = fmt.Sprintf("%d", constants.KubeAPIServerPort)
		portKCM       = fmt.Sprintf("%d", constants.KubeControllerManagerPort)
		portScheduler = fmt.Sprintf("%d", constants.KubeSchedulerPort)

		errs   []error
		result []controlPlaneComponent
	)

	type componentConfig struct {
		name        string
		podKey      string
		args        []string
		defaultAddr string
		defaultPort string
		endpoint    string
	}

	components := []componentConfig{
		{
			name:        "kube-apiserver",
			podKey:      constants.KubeAPIServer,
			args:        []string{argAdvertiseAddress, argPort},
			defaultAddr: addressAPIServer,
			defaultPort: portAPIServer,
			endpoint:    endpointLivez,
		},
		{
			name:        "kube-controller-manager",
			podKey:      constants.KubeControllerManager,
			args:        []string{argBindAddress, argPort},
			defaultAddr: addressKCM,
			defaultPort: portKCM,
			endpoint:    endpointHealthz,
		},
		{
			name:        "kube-scheduler",
			podKey:      constants.KubeScheduler,
			args:        []string{argBindAddress, argPort},
			defaultAddr: addressScheduler,
			defaultPort: portScheduler,
			endpoint:    endpointLivez,
		},
	}

	for _, component := range components {
		address, port := component.defaultAddr, component.defaultPort

		values, err := getControlPlaneComponentAddressAndPort(
			podMap[component.podKey],
			component.podKey,
			component.args,
		)
		if err != nil {
			errs = append(errs, err)
		}

		if len(values[0]) != 0 {
			address = values[0]
		}
		if len(values[1]) != 0 {
			port = values[1]
		}

		result = append(result, controlPlaneComponent{
			name: component.name,
			url:  fmt.Sprintf("https://%s/%s", net.JoinHostPort(address, port), component.endpoint),
		})
	}

	if len(errs) > 0 {
		return nil, utilerrors.NewAggregate(errs)
	}
	return result, nil
}

// WaitForControlPlaneComponents waits for all control plane components to report "ok".
func (w *KubeWaiter) WaitForControlPlaneComponents(podMap map[string]*v1.Pod, apiSeverAddress string) error {
	fmt.Printf("[control-plane-check] Waiting for healthy control plane components."+
		" This can take up to %v\n", w.timeout)

	components, err := getControlPlaneComponents(podMap, apiSeverAddress)
	if err != nil {
		return errors.Wrap(err, "could not parse the address and port of all control plane components")
	}

	var errs []error
	errChan := make(chan error, len(components))

	for _, comp := range components {
		fmt.Printf("[control-plane-check] Checking %s at %s\n", comp.name, comp.url)

		go func(comp controlPlaneComponent) {
			tr := &http.Transport{
				TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
			}
			client := &http.Client{Transport: tr}
			start := time.Now()
			var lastError error

			err := wait.PollUntilContextTimeout(
				context.Background(),
				constants.KubernetesAPICallRetryInterval,
				w.timeout,
				true, func(ctx context.Context) (bool, error) {
					resp, err := client.Get(comp.url)
					if err != nil {
						lastError = errors.WithMessagef(err, "%s check failed at %s", comp.name, comp.url)
						return false, nil
					}

					defer func() {
						_ = resp.Body.Close()
					}()
					if resp.StatusCode != http.StatusOK {
						lastError = errors.Errorf("%s check failed at %s with status: %d",
							comp.name, comp.url, resp.StatusCode)
						return false, nil
					}

					return true, nil
				})
			if err != nil {
				fmt.Printf("[control-plane-check] %s is not healthy after %v\n", comp.name, time.Since(start))
				errChan <- lastError
				return
			}
			fmt.Printf("[control-plane-check] %s is healthy after %v\n", comp.name, time.Since(start))
			errChan <- nil
		}(comp)
	}

	for i := 0; i < len(components); i++ {
		if err := <-errChan; err != nil {
			errs = append(errs, err)
		}
	}
	return utilerrors.NewAggregate(errs)
}

// WaitForAPI waits for the API Server's /healthz endpoint to report "ok"
func (w *KubeWaiter) WaitForAPI() error {
	fmt.Printf("[api-check] Waiting for a healthy API server. This can take up to %v\n", w.timeout)

	start := time.Now()
	err := wait.PollUntilContextTimeout(
		context.Background(),
		constants.KubernetesAPICallRetryInterval,
		w.timeout,
		true, func(ctx context.Context) (bool, error) {
			healthStatus := 0
			w.client.Discovery().RESTClient().Get().AbsPath("/healthz").Do(ctx).StatusCode(&healthStatus)
			if healthStatus != http.StatusOK {
				return false, nil
			}
			return true, nil
		})
	if err != nil {
		fmt.Printf("[api-check] The API server is not healthy after %v\n", time.Since(start))
		return err
	}

	fmt.Printf("[api-check] The API server is healthy after %v\n", time.Since(start))
	return nil
}

// WaitForPodsWithLabel will lookup pods with the given label and wait until they are all
// reporting status as running.
func (w *KubeWaiter) WaitForPodsWithLabel(kvLabel string) error {

	lastKnownPodNumber := -1
	return wait.PollUntilContextTimeout(context.Background(),
		constants.KubernetesAPICallRetryInterval, w.timeout,
		true, func(_ context.Context) (bool, error) {
			listOpts := metav1.ListOptions{LabelSelector: kvLabel}
			pods, err := w.client.CoreV1().Pods(metav1.NamespaceSystem).List(context.TODO(), listOpts)
			if err != nil {
				fmt.Fprintf(w.writer, "[apiclient] Error getting Pods with label selector %q [%v]\n", kvLabel, err)
				return false, nil
			}

			if lastKnownPodNumber != len(pods.Items) {
				fmt.Fprintf(w.writer, "[apiclient] Found %d Pods for label selector %s\n", len(pods.Items), kvLabel)
				lastKnownPodNumber = len(pods.Items)
			}

			if len(pods.Items) == 0 {
				return false, nil
			}

			for _, pod := range pods.Items {
				if pod.Status.Phase != v1.PodRunning {
					return false, nil
				}
			}

			return true, nil
		})
}

// WaitForKubelet blocks until the kubelet /healthz endpoint returns 'ok'.
func (w *KubeWaiter) WaitForKubelet(healthzAddress string, healthzPort int32) error {
	var (
		lastError       error
		start           = time.Now()
		healthzEndpoint = fmt.Sprintf("http://%s:%d/healthz", healthzAddress, healthzPort)
	)

	if healthzPort == 0 {
		fmt.Println("[kubelet-check] Skipping the kubelet health check because the healthz port is set to 0")
		return nil
	}
	fmt.Printf("[kubelet-check] Waiting for a healthy kubelet at %s. This can take up to %v\n",
		healthzEndpoint, w.timeout)

	formatError := func(cause string) error {
		return errors.Errorf("The HTTP call equal to 'curl -sSL %s' returned %s\n",
			healthzEndpoint, cause)
	}

	err := wait.PollUntilContextTimeout(
		context.Background(),
		constants.KubernetesAPICallRetryInterval,
		w.timeout,
		true, func(ctx context.Context) (bool, error) {
			client := &http.Client{Transport: netutil.SetOldTransportDefaults(&http.Transport{})}
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthzEndpoint, nil)
			if err != nil {
				lastError = formatError(fmt.Sprintf("error: %v", err))
				return false, err
			}
			resp, err := client.Do(req)
			if err != nil {
				lastError = formatError(fmt.Sprintf("error: %v", err))
				return false, nil
			}
			defer func() {
				_ = resp.Body.Close()
			}()
			if resp.StatusCode != http.StatusOK {
				lastError = formatError(fmt.Sprintf("status code: %d", resp.StatusCode))
				return false, nil
			}

			return true, nil
		})
	if err != nil {
		fmt.Printf("[kubelet-check] The kubelet is not healthy after %v\n", time.Since(start))
		return lastError
	}

	fmt.Printf("[kubelet-check] The kubelet is healthy after %v\n", time.Since(start))
	return nil
}

// SetTimeout adjusts the timeout to the specified duration
func (w *KubeWaiter) SetTimeout(timeout time.Duration) {
	w.timeout = timeout
}

// WaitForStaticPodControlPlaneHashes blocks until it timeouts or gets a hash map for all components and their Static Pods
func (w *KubeWaiter) WaitForStaticPodControlPlaneHashes(nodeName string) (map[string]string, error) {

	componentHash := ""
	var err, lastErr error
	mirrorPodHashes := map[string]string{}
	for _, component := range constants.ControlPlaneComponents {
		err = wait.PollUntilContextTimeout(context.Background(),
			constants.KubernetesAPICallRetryInterval, w.timeout,
			true, func(_ context.Context) (bool, error) {
				componentHash, err = getStaticPodSingleHash(w.client, nodeName, component)
				if err != nil {
					lastErr = err
					return false, nil
				}
				return true, nil
			})
		if err != nil {
			return nil, lastErr
		}
		mirrorPodHashes[component] = componentHash
	}

	return mirrorPodHashes, nil
}

// WaitForStaticPodSingleHash blocks until it timeouts or gets a hash for a single component and its Static Pod
func (w *KubeWaiter) WaitForStaticPodSingleHash(nodeName string, component string) (string, error) {

	componentPodHash := ""
	var err, lastErr error
	err = wait.PollUntilContextTimeout(context.Background(),
		constants.KubernetesAPICallRetryInterval, w.timeout,
		true, func(_ context.Context) (bool, error) {
			componentPodHash, err = getStaticPodSingleHash(w.client, nodeName, component)
			if err != nil {
				lastErr = err
				return false, nil
			}
			return true, nil
		})

	if err != nil {
		err = lastErr
	}
	return componentPodHash, err
}

// WaitForStaticPodHashChange blocks until it timeouts or notices that the Mirror Pod (for the Static Pod, respectively) has changed
// This implicitly means this function blocks until the kubelet has restarted the Static Pod in question
func (w *KubeWaiter) WaitForStaticPodHashChange(nodeName, component, previousHash string) error {
	var err, lastErr error
	err = wait.PollUntilContextTimeout(context.Background(),
		constants.KubernetesAPICallRetryInterval, w.timeout,
		true, func(_ context.Context) (bool, error) {
			hash, err := getStaticPodSingleHash(w.client, nodeName, component)
			if err != nil {
				lastErr = err
				return false, nil
			}
			// Set lastErr to nil to be able to later distinguish between getStaticPodSingleHash() and timeout errors
			lastErr = nil
			// We should continue polling until the UID changes
			if hash == previousHash {
				return false, nil
			}

			return true, nil
		})

	// If lastError is not nil, this must be a getStaticPodSingleHash() error, else if err is not nil there was a poll timeout
	if lastErr != nil {
		return lastErr
	}
	return errors.Wrapf(err, "static Pod hash for component %s on Node %s did not change after %v", component, nodeName, w.timeout)
}

// getStaticPodSingleHash computes hashes for a single Static Pod resource
func getStaticPodSingleHash(client clientset.Interface, nodeName string, component string) (string, error) {

	staticPodName := fmt.Sprintf("%s-%s", component, nodeName)
	staticPod, err := client.CoreV1().Pods(metav1.NamespaceSystem).Get(context.TODO(), staticPodName, metav1.GetOptions{})
	if err != nil {
		return "", errors.Wrapf(err, "failed to obtain static Pod hash for component %s on Node %s", component, nodeName)
	}

	staticPodHash := staticPod.Annotations["kubernetes.io/config.hash"]
	return staticPodHash, nil
}

// PrintControlPlaneErrorHelpScreen prints help text on wait ControlPlane components errors.
func PrintControlPlaneErrorHelpScreen(outputWriter io.Writer, criSocket string) {
	context := struct {
		Socket string
	}{
		Socket: criSocket,
	}
	_ = controlPlaneFailTempl.Execute(outputWriter, context)
	fmt.Println("")
}

// PrintKubeletErrorHelpScreen prints help text on kubelet errors.
func PrintKubeletErrorHelpScreen(outputWriter io.Writer) {
	fmt.Fprintln(outputWriter, kubeletFailMsg)
}
