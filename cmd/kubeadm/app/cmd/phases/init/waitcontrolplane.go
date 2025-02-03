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

package phases

import (
	"fmt"
	"io"
	"text/template"
	"time"

	"github.com/lithammer/dedent"
	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeletconfig "k8s.io/kubelet/config/v1beta1"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

var (
	kubeletFailTempl = template.Must(template.New("init").Parse(dedent.Dedent(`
	Unfortunately, an error has occurred:
		{{ .Error }}

	This error is likely caused by:
		- The kubelet is not running
		- The kubelet is unhealthy due to a misconfiguration of the node in some way (required cgroups disabled)

	If you are on a systemd-powered system, you can try to troubleshoot the error with the following commands:
		- 'systemctl status kubelet'
		- 'journalctl -xeu kubelet'

	Additionally, a control plane component may have crashed or exited when started by the container runtime.
	To troubleshoot, list all containers using your preferred container runtimes CLI.
	Here is one example how you may list all running Kubernetes containers by using crictl:
		- 'crictl --runtime-endpoint {{ .Socket }} ps -a | grep kube | grep -v pause'
		Once you have found the failing container, you can inspect its logs with:
		- 'crictl --runtime-endpoint {{ .Socket }} logs CONTAINERID'
	`)))
)

// NewWaitControlPlanePhase is a hidden phase that runs after the control-plane and etcd phases
func NewWaitControlPlanePhase() workflow.Phase {
	phase := workflow.Phase{
		Name:  "wait-control-plane",
		Short: "Wait for the control plane to start",
		// TODO: unhide this phase once WaitForAllControlPlaneComponents goes GA:
		// https://github.com/kubernetes/kubeadm/issues/2907
		Hidden: true,
		Run:    runWaitControlPlanePhase,
	}
	return phase
}

func runWaitControlPlanePhase(c workflow.RunData) error {
	data, ok := c.(InitData)
	if !ok {
		return errors.New("wait-control-plane phase invoked with an invalid data struct")
	}

	// If we're dry-running, print the generated manifests.
	// TODO: think of a better place to move this call - e.g. a hidden phase.
	if data.DryRun() {
		if err := dryrunutil.PrintFilesIfDryRunning(true /* needPrintManifest */, data.ManifestDir(), data.OutputWriter()); err != nil {
			return errors.Wrap(err, "error printing files on dryrun")
		}
	}

	// Both Wait* calls below use a /healthz endpoint, thus a client without permissions works fine
	client, err := data.ClientWithoutBootstrap()
	if err != nil {
		return errors.Wrap(err, "cannot obtain client without bootstrap")
	}

	waiter, err := newControlPlaneWaiter(data.DryRun(), 0, client, data.OutputWriter())
	if err != nil {
		return errors.Wrap(err, "error creating waiter")
	}

	fmt.Printf("[wait-control-plane] Waiting for the kubelet to boot up the control plane as static Pods"+
		" from directory %q\n",
		data.ManifestDir())

	handleError := func(err error) error {
		context := struct {
			Error  string
			Socket string
		}{
			Error:  fmt.Sprintf("%v", err),
			Socket: data.Cfg().NodeRegistration.CRISocket,
		}

		kubeletFailTempl.Execute(data.OutputWriter(), context)
		return errors.New("could not initialize a Kubernetes cluster")
	}

	waiter.SetTimeout(data.Cfg().Timeouts.KubeletHealthCheck.Duration)
	kubeletConfig := data.Cfg().ClusterConfiguration.ComponentConfigs[componentconfigs.KubeletGroup].Get()
	kubeletConfigTyped, ok := kubeletConfig.(*kubeletconfig.KubeletConfiguration)
	if !ok {
		return errors.New("could not convert the KubeletConfiguration to a typed object")
	}
	if err := waiter.WaitForKubelet(kubeletConfigTyped.HealthzBindAddress, *kubeletConfigTyped.HealthzPort); err != nil {
		return handleError(err)
	}

	var podMap map[string]*v1.Pod
	waiter.SetTimeout(data.Cfg().Timeouts.ControlPlaneComponentHealthCheck.Duration)
	if features.Enabled(data.Cfg().ClusterConfiguration.FeatureGates, features.WaitForAllControlPlaneComponents) {
		podMap, err = staticpodutil.ReadMultipleStaticPodsFromDisk(data.ManifestDir(),
			constants.ControlPlaneComponents...)
		if err == nil {
			err = waiter.WaitForControlPlaneComponents(podMap,
				data.Cfg().LocalAPIEndpoint.AdvertiseAddress)
		}
	} else {
		err = waiter.WaitForAPI()
	}
	if err != nil {
		return handleError(err)
	}

	return nil
}

// newControlPlaneWaiter returns a new waiter that is used to wait on the control plane to boot up.
func newControlPlaneWaiter(dryRun bool, timeout time.Duration, client clientset.Interface, out io.Writer) (apiclient.Waiter, error) {
	if dryRun {
		return dryrunutil.NewWaiter(), nil
	}

	return apiclient.NewKubeWaiter(client, timeout, out), nil
}
