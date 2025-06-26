/*
Copyright 2024 The Kubernetes Authors.

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
	"io"
	"time"

	clientset "k8s.io/client-go/kubernetes"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

// NewWaitControlPlanePhase is a hidden phase that runs after the control-plane and etcd phases
func NewWaitControlPlanePhase() workflow.Phase {
	phase := workflow.Phase{
		Name:  "wait-control-plane",
		Short: "Wait for the control plane to start",
		Run:   runWaitControlPlanePhase,
	}
	return phase
}

func runWaitControlPlanePhase(c workflow.RunData) error {
	data, ok := c.(JoinData)
	if !ok {
		return errors.New("wait-control-plane phase invoked with an invalid data struct")
	}

	if data.Cfg().ControlPlane == nil {
		return nil
	}

	client, err := data.Client()
	if err != nil {
		return err
	}

	waiter, err := newControlPlaneWaiter(data.DryRun(), 0, client, data.OutputWriter())
	if err != nil {
		return errors.Wrap(err, "error creating waiter")
	}

	waiter.SetTimeout(data.Cfg().Timeouts.ControlPlaneComponentHealthCheck.Duration)
	pods, err := staticpodutil.ReadMultipleStaticPodsFromDisk(data.ManifestDir(),
		constants.ControlPlaneComponents...)
	if err != nil {
		return err
	}
	if err = waiter.WaitForControlPlaneComponents(pods,
		data.Cfg().ControlPlane.LocalAPIEndpoint.AdvertiseAddress); err != nil {
		apiclient.PrintControlPlaneErrorHelpScreen(data.OutputWriter(), data.Cfg().NodeRegistration.CRISocket)
		return errors.Wrap(err, "failed while waiting for the control plane to start")
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
