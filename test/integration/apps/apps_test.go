/*
Copyright 2017 The Kubernetes Authors.

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
	"log"
	"testing"

	v1 "k8s.io/api/core/v1"
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/integration/apps/core"
	"k8s.io/kubernetes/test/integration/apps/hrcontrollers"
	"k8s.io/kubernetes/test/integration/apps/podconfig"
	"k8s.io/kubernetes/test/integration/apps/podcontrollers"
	"k8s.io/kubernetes/test/integration/apps/upgrades"
	controlplane "k8s.io/kubernetes/test/integration/fixtures/controlplane"
	"k8s.io/kubernetes/test/integration/framework"
)

//RunBySuite Below flag will be set to 'True' if run the by the script via go test -ldflags
var RunBySuite string

var Tests = []testing.InternalTest{
	{Name: core.Name, F: core.RunTests},
	{Name: podconfig.Name, F: podconfig.RunTests},
	{Name: podcontrollers.Name, F: podcontrollers.RunTests},
	{Name: hrcontrollers.Name, F: hrcontrollers.RunTests},
	{Name: upgrades.Name, F: upgrades.RunTests},
}

var CP *controlplane.ControlPlane

func TestMain(m *testing.M) {
	flag.Parse()
	framework.EtcdMain(m.Run)
}

func Setup(t *testing.T) error {
	//Wait for kube components
	CP = controlplane.NewControlPlane("Apps")
	CP.Start(t)
	return nil
}

func TearDown(t *testing.T) error {

	CP.TearDown(t)
	return nil
}

func TestApps(t *testing.T) {

	Setup(t)
	defer TearDown(t)

	NSOpt := meta_v1.ListOptions{}
	NS := v1.Namespace{}
	NS.Name = "testing"
	NS.ObjectMeta.Name = "testing"

	ns, err := CP.Cli.Core().Namespaces().Create(&NS)

	if err != nil {
		t.Fatalf("Error creating namespace =%v", err)
	}

	log.Printf("Namespance created is %v", ns)

	nsList, err := CP.Cli.Core().Namespaces().List(NSOpt)
	if err != nil {
		t.Fatalf("Error Listing namespace =%v", err)
	}

	log.Printf("There are %d Namespaces", len(nsList.Items))
	for _, n := range nsList.Items {
		log.Printf("%s", n.Name)
	}

	for _, tst := range Tests {
		t.Run(tst.Name, tst.F)
	}

}
