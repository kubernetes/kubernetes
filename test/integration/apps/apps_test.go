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
	"testing"

	common "k8s.io/kubernetes/test/integration/apps/common_types"
	"k8s.io/kubernetes/test/integration/apps/core"
	"k8s.io/kubernetes/test/integration/apps/hrcontrollers"
	"k8s.io/kubernetes/test/integration/apps/podconfig"
	"k8s.io/kubernetes/test/integration/apps/podcontrollers"
	"k8s.io/kubernetes/test/integration/apps/upgrades"
	"k8s.io/kubernetes/test/integration/framework"
	"time"
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

func TestMain(m *testing.M) {
	flag.Parse()
	framework.EtcdMain(m.Run)
}

func Setup(t *testing.T) error {
	//Wait for kube components
	common.Initialize(t)
	return nil
}

func TearDown(t *testing.T) error {

	//t.Logf("Finsiesed TearDown\n")
	return nil
}

func TestApps(t *testing.T) {

	if RunBySuite != "True" {
		t.Skipf("Sikkping the tests...only run by the script")
		t.SkipNow()
	}

	//Setup
	Setup(t)
	defer TearDown(t)

	for _, tst := range Tests {
		t.Run(tst.Name, tst.F)
	}

	time.Sleep(time.Hour)

}
