/*
Copyright 2019 The Kubernetes Authors.

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

package client

import (
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"regexp"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	utildbus "k8s.io/kubernetes/pkg/util/dbus"
	"k8s.io/kubernetes/pkg/util/iptables"
	"k8s.io/kubernetes/test/integration/framework"
	utilsexec "k8s.io/utils/exec"
)

func TestReconnectBrokenTCP(t *testing.T) {
	// TODO: remove env setting when kubernetes/client-go#374 is fixed
	// use http to detect non-responding tcp connection
	defer func() {
		if err := os.Setenv("DISABLE_HTTP2", ""); err != nil {
			t.Fatalf("failed to re-enable http2: %v", err)
		}
	}()
	if err := os.Setenv("DISABLE_HTTP2", "true"); err != nil {
		t.Fatalf("failed to disable http2: %v", err)
	}

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig).CoreV1().Endpoints("default")
	w, err := client.Watch(metav1.ListOptions{ResourceVersion: "0"})
	if err != nil {
		t.Fatalf("failed to watch endpoints: %v", err)
	}

	// find port that talks to test server
	connections, err := exec.Command("ss", "-tn").Output()
	if err != nil {
		t.Fatalf("failed ot list tcp connections: %v", err)
	}

	u, err := url.Parse(server.ClientConfig.Host)
	if err != nil {
		t.Fatalf("failed to parse url %s: %v", server.ClientConfig.Host, err)
	}
	pattern := regexp.MustCompile(fmt.Sprintf(`%s[\t ]+[0-9.]+:([0-9]+)`, u.Host))
	ports := pattern.FindStringSubmatch(string(connections))
	sport := ports[1]

	execer := utilsexec.New()
	dbus := utildbus.New()
	ipt := iptables.New(execer, dbus, iptables.ProtocolIpv4)
	ip6t := iptables.New(execer, dbus, iptables.ProtocolIpv6)

	// drop tcp traffic from the port opened by watch to test server
	defer func() {
		if err := ipt.DeleteRule(iptables.TableFilter, iptables.ChainOutput, "-p", "tcp", "--jump", "DROP", "--sport", sport); err != nil {
			t.Fatalf("failed to delete firewall: %v", err)
		}
		if err := ip6t.DeleteRule(iptables.TableFilter, iptables.ChainOutput, "-p", "tcp", "--jump", "DROP", "--sport", sport); err != nil {
			t.Fatalf("failed to delete ipv6 firewall: %v", err)
		}
	}()
	if _, err := ipt.EnsureRule(iptables.Prepend, iptables.TableFilter, iptables.ChainOutput, "-p", "tcp", "--jump", "DROP", "--sport", sport); err != nil {
		t.Fatalf("failed to add firewall: %v", err)
	}
	if _, err := ip6t.EnsureRule(iptables.Prepend, iptables.TableFilter, iptables.ChainOutput, "-p", "tcp", "--jump", "DROP", "--sport", sport); err != nil {
		t.Fatalf("failed to add ipv6 firewall: %v", err)
	}

	// create three endpoints
	expected := 3
	go func() {
		for i := 0; i < expected; i++ {
			if _, err := client.Create(&v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name: fmt.Sprintf("endpoints-%d", i),
				},
			}); err != nil {
				t.Fatalf("failed to create endpoints: %v", err)
			}
		}
	}()

	// expect to observe three endpoints creation watch events
	stopTimer := time.NewTimer(10 * time.Second)
	defer stopTimer.Stop()
	count := 0
	running := true
	for running {
		select {
		case got, ok := <-w.ResultChan():
			if !ok {
				t.Fatal("watch closed unexpectedly")
			}
			if e, a := watch.Added, got.Type; e != a {
				t.Errorf("wanted %v, got %v", e, a)
			}
			switch o := got.Object.(type) {
			case *v1.Endpoints:
				if e, a := fmt.Sprintf("endpoints-%d", count), o.Name; e != a {
					t.Errorf("wanted %v, got %v", e, a)
				}
			default:
				t.Fatalf("unexpected watch event containing object %#q", got)
			}
			count++
		case <-stopTimer.C:
			running = false
		}
	}
	if count != expected {
		t.Fatalf("failed to observe watch events within timeout, expected: %d, got: %d", expected, count)
	}
}
