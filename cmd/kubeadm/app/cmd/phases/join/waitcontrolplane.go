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

	"github.com/pkg/errors"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

// NewWaitControlPlanePhase is a hidden phase that runs after the control-plane and etcd phases
func NewWaitControlPlanePhase() workflow.Phase {
	phase := workflow.Phase{
		Name: "wait-control-plane",
		// TODO: remove this EXPERIMENTAL prefix once WaitForAllControlPlaneComponents goes GA:
		// https://github.com/kubernetes/kubeadm/issues/2907
		Short: "EXPERIMENTAL: Wait for the control plane to start",
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

	initCfg, err := data.InitCfg()
	if err != nil {
		return errors.Wrap(err, "could not obtain InitConfiguration during the wait-control-plane phase")
	}

	// TODO: remove this check once WaitForAllControlPlaneComponents goes GA
	// https://github.com/kubernetes/kubeadm/issues/2907
	if !features.Enabled(initCfg.ClusterConfiguration.FeatureGates, features.WaitForAllControlPlaneComponents) {
		klog.V(5).Infof("[wait-control-plane] Skipping phase as the feature gate WaitForAllControlPlaneComponents is disabled")
		return nil
	}

	waiter, err := newControlPlaneWaiter(data.DryRun(), 0, nil, data.OutputWriter())
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
		return err
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
