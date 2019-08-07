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

package lifecycle

import (
	"fmt"
	"os/exec"
	"path"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/version"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

// RealVersion turns a version constants into a version string deployable on
// GKE.  See hack/get-build.sh for more information.
func RealVersion(s string) (string, error) {
	e2elog.Logf("Getting real version for %q", s)
	v, _, err := framework.RunCmd(path.Join(framework.TestContext.RepoRoot, "hack/get-build.sh"), "-v", s)
	if err != nil {
		return v, fmt.Errorf("error getting real version for %q: %v", s, err)
	}
	e2elog.Logf("Version for %q is %q", s, v)
	return strings.TrimPrefix(strings.TrimSpace(v), "v"), nil
}

func traceRouteToMaster() {
	traceroute, err := exec.LookPath("traceroute")
	if err != nil {
		e2elog.Logf("Could not find traceroute program")
		return
	}
	cmd := exec.Command(traceroute, "-I", framework.GetMasterHost())
	out, err := cmd.Output()
	if len(out) != 0 {
		e2elog.Logf(string(out))
	}
	if exiterr, ok := err.(*exec.ExitError); err != nil && ok {
		e2elog.Logf("Error while running traceroute: %s", exiterr.Stderr)
	}
}

// CheckMasterVersion validates the master version
func CheckMasterVersion(c clientset.Interface, want string) error {
	e2elog.Logf("Checking master version")
	var err error
	var v *version.Info
	waitErr := wait.PollImmediate(5*time.Second, 2*time.Minute, func() (bool, error) {
		v, err = c.Discovery().ServerVersion()
		if err != nil {
			traceRouteToMaster()
			return false, nil
		}
		return true, nil
	})
	if waitErr != nil {
		return fmt.Errorf("CheckMasterVersion() couldn't get the master version: %v", err)
	}
	// We do prefix trimming and then matching because:
	// want looks like:  0.19.3-815-g50e67d4
	// got  looks like: v0.19.3-815-g50e67d4034e858-dirty
	got := strings.TrimPrefix(v.GitVersion, "v")
	if !strings.HasPrefix(got, want) {
		return fmt.Errorf("master had kube-apiserver version %s which does not start with %s", got, want)
	}
	e2elog.Logf("Master is at version %s", want)
	return nil
}

// CheckNodesVersions validates the nodes versions
func CheckNodesVersions(cs clientset.Interface, want string) error {
	l := framework.GetReadySchedulableNodesOrDie(cs)
	for _, n := range l.Items {
		// We do prefix trimming and then matching because:
		// want   looks like:  0.19.3-815-g50e67d4
		// kv/kvp look  like: v0.19.3-815-g50e67d4034e858-dirty
		kv, kpv := strings.TrimPrefix(n.Status.NodeInfo.KubeletVersion, "v"),
			strings.TrimPrefix(n.Status.NodeInfo.KubeProxyVersion, "v")
		if !strings.HasPrefix(kv, want) {
			return fmt.Errorf("node %s had kubelet version %s which does not start with %s",
				n.ObjectMeta.Name, kv, want)
		}
		if !strings.HasPrefix(kpv, want) {
			return fmt.Errorf("node %s had kube-proxy version %s which does not start with %s",
				n.ObjectMeta.Name, kpv, want)
		}
	}
	return nil
}
