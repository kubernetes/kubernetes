/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/cmd/uber-cluster-controller/app/options"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/controller/cluster"
)

func Run(c *options.ClusterController) error {
	uberConfig, err := clientcmd.BuildConfigFromFlags(c.Ubernetes, "")
	if err != nil {
		return err
	}
	uberClient, err := client.New(uberConfig)
	if err != nil {
		return err
	}
	cc := cluster.New(uberClient)
	cc.Run()
	return nil
}
