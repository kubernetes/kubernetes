/*
Copyright 2016 The Kubernetes Authors.

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

package app

import (
	"fmt"
	"os"

	"k8s.io/kubernetes/federation/pkg/kubefed"
	_ "k8s.io/kubernetes/pkg/client/metrics/prometheus" // for client metric registration
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/logs"
	"k8s.io/kubernetes/pkg/version"
	_ "k8s.io/kubernetes/pkg/version/prometheus" // for version metric registration
)

const (
	hyperkubeImageName = "gcr.io/google_containers/hyperkube-amd64"
	DefaultEtcdImage   = "gcr.io/google_containers/etcd:3.0.17"
)

func GetDefaultServerImage() string {
	return fmt.Sprintf("%s:%s", hyperkubeImageName, version.Get())
}

func Run() error {
	logs.InitLogs()
	defer logs.FlushLogs()

	cmd := kubefed.NewKubeFedCommand(cmdutil.NewFactory(nil), os.Stdin, os.Stdout, os.Stderr, GetDefaultServerImage(), DefaultEtcdImage)
	return cmd.Execute()
}
