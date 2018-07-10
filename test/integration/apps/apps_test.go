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

package apps

import (
	"flag"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/integration/apps/upgrades"
	controlplane "k8s.io/kubernetes/test/integration/fixtures/controlplane"
	"k8s.io/kubernetes/test/integration/framework"
)

//RunBySuite Below flag will be set to 'True' if run the by the script via go test -ldflags
var RunBySuite string

var Tests = []controlplane.Tests{
	{Name: upgrades.TestPkgName, F: upgrades.RunTests},
}

var controlPlane *controlplane.ControlPlane

func TestMain(m *testing.M) {
	flag.Parse()
	framework.EtcdMain(m.Run)
}

func Setup(t *testing.T) error {
	//Wait for kube components
	controlPlane = controlplane.NewControlPlane("Apps")
	controlPlane.Start(t)

	//Setup a testing namespace
	ns := v1.Namespace{}
	ns.Name = "testing"
	ns.ObjectMeta.Name = "testing"

	_, err := controlPlane.Client.Core().Namespaces().Create(&ns)
	if err != nil {
		t.Fatalf("Error creating namespace =%v", err)
	}

	return nil
}

func TearDown(t *testing.T) error {

	controlPlane.TearDown(t)
	return nil
}

func TestApps(t *testing.T) {

	Setup(t)
	defer TearDown(t)

	for _, tst := range Tests {
		t.Run(tst.Name, func(t *testing.T) {
			tst.F(t, controlPlane)
		})
	}
}
