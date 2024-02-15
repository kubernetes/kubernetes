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

package util

import (
	"context"
	"fmt"

	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/options"
	cloudctrlmgrtesting "k8s.io/cloud-provider/app/testing"
	kubectrlmgrtesting "k8s.io/kubernetes/cmd/kube-controller-manager/app/testing"
	kubeschedulertesting "k8s.io/kubernetes/cmd/kube-scheduler/app/testing"
)

type ComponentTester interface {
	StartTestServer(ctx context.Context, customFlags []string) (*options.SecureServingOptionsWithLoopback, *server.SecureServingInfo, func(), error)
}

type kubeControllerManagerTester struct {
	controllers string
}

func NewKubeControllerManagerTester(controllers string) ComponentTester {
	return &kubeControllerManagerTester{
		controllers: controllers,
	}
}
func (k *kubeControllerManagerTester) StartTestServer(ctx context.Context, customFlags []string) (*options.SecureServingOptionsWithLoopback, *server.SecureServingInfo, func(), error) {
	// avoid starting any controller loops, we're just testing serving
	customFlags = append([]string{fmt.Sprintf("--controllers=%s", k.controllers)}, customFlags...)
	gotResult, err := kubectrlmgrtesting.StartTestServer(ctx, customFlags)
	if err != nil {
		return nil, nil, nil, err
	}
	return gotResult.Options.SecureServing, gotResult.Config.SecureServing, gotResult.TearDownFn, err
}

type cloudControllerManagerTester struct{}

func NewCloudControllerManagerTester() ComponentTester {
	return cloudControllerManagerTester{}
}

func (cloudControllerManagerTester) StartTestServer(ctx context.Context, customFlags []string) (*options.SecureServingOptionsWithLoopback, *server.SecureServingInfo, func(), error) {
	gotResult, err := cloudctrlmgrtesting.StartTestServer(ctx, customFlags)
	if err != nil {
		return nil, nil, nil, err
	}
	return gotResult.Options.SecureServing, gotResult.Config.SecureServing, gotResult.TearDownFn, err
}

type kubeSchedulerTester struct{}

func NewKubeSchedulerTester() ComponentTester {
	return kubeSchedulerTester{}
}

func (kubeSchedulerTester) StartTestServer(ctx context.Context, customFlags []string) (*options.SecureServingOptionsWithLoopback, *server.SecureServingInfo, func(), error) {
	gotResult, err := kubeschedulertesting.StartTestServer(ctx, customFlags)
	if err != nil {
		return nil, nil, nil, err
	}
	return gotResult.Options.SecureServing, gotResult.Config.SecureServing, gotResult.TearDownFn, err
}
