/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package vagrant_cloud

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// startSaltTestServer starts a test server that mocks the Salt REST API
func startSaltTestServer() *httptest.Server {

	// mock responses
	var (
		testSaltMinionsResponse = []byte(`{ "return": [{"kubernetes-minion-1": {"kernel": "Linux", "domain": "", "zmqversion": "3.2.4", "kernelrelease": "3.11.10-301.fc20.x86_64", "pythonpath": ["/usr/bin", "/usr/lib64/python27.zip", "/usr/lib64/python2.7", "/usr/lib64/python2.7/plat-linux2", "/usr/lib64/python2.7/lib-tk", "/usr/lib64/python2.7/lib-old", "/usr/lib64/python2.7/lib-dynload", "/usr/lib64/python2.7/site-packages", "/usr/lib/python2.7/site-packages"], "etcd_servers": "10.245.1.2", "ip_interfaces": {"lo": ["127.0.0.1"], "docker0": ["172.17.42.1"], "enp0s8": ["10.245.2.2"], "p2p1": ["10.0.2.15"]}, "shell": "/bin/sh", "mem_total": 491, "saltversioninfo": [2014, 1, 7], "osmajorrelease": ["20"], "node_ip": "10.245.2.2", "id": "kubernetes-minion-1", "osrelease": "20", "ps": "ps -efH", "server_id": 1005530826, "num_cpus": 1, "hwaddr_interfaces": {"lo": "00:00:00:00:00:00", "docker0": "56:84:7a:fe:97:99", "enp0s8": "08:00:27:17:c5:0f", "p2p1": "08:00:27:96:96:e1"}, "virtual": "VirtualBox", "osfullname": "Fedora", "master": "kubernetes-master", "ipv4": ["10.0.2.15", "10.245.2.2", "127.0.0.1", "172.17.42.1"], "ipv6": ["::1", "fe80::a00:27ff:fe17:c50f", "fe80::a00:27ff:fe96:96e1"], "cpu_flags": ["fpu", "vme", "de", "pse", "tsc", "msr", "pae", "mce", "cx8", "apic", "sep", "mtrr", "pge", "mca", "cmov", "pat", "pse36", "clflush", "mmx", "fxsr", "sse", "sse2", "syscall", "nx", "rdtscp", "lm", "constant_tsc", "rep_good", "nopl", "pni", "monitor", "ssse3", "lahf_lm"], "localhost": "kubernetes-minion-1", "lsb_distrib_id": "Fedora", "fqdn_ip4": ["127.0.0.1"], "fqdn_ip6": [], "nodename": "kubernetes-minion-1", "saltversion": "2014.1.7", "saltpath": "/usr/lib/python2.7/site-packages/salt", "pythonversion": [2, 7, 5, "final", 0], "host": "kubernetes-minion-1", "os_family": "RedHat", "oscodename": "Heisenbug", "defaultencoding": "UTF-8", "osfinger": "Fedora-20", "roles": ["kubernetes-pool"], "num_gpus": 1, "cpu_model": "Intel(R) Core(TM) i7-4600U CPU @ 2.10GHz", "fqdn": "kubernetes-minion-1", "osarch": "x86_64", "cpuarch": "x86_64", "gpus": [{"model": "VirtualBox Graphics Adapter", "vendor": "unknown"}], "path": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin", "os": "Fedora", "defaultlanguage": "en_US"}}]}`)
		testSaltLoginResponse   = []byte(`{ "return": [{"perms": [".*"], "start": 1407355696.564397, "token": "ca74fa1c48ce40e204a1e820d2fa14b7cf033137", "expire": 1407398896.564398, "user": "vagrant", "eauth": "pam"}]}`)
		testSaltFailure         = []byte(`failure`)
	)

	handler := func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case "GET":
			switch r.URL.Path {
			case "/minions":
				w.Write(testSaltMinionsResponse)
				return
			}
		case "POST":
			switch r.URL.Path {
			case "/login":
				w.Write(testSaltLoginResponse)
				return
			}
		}
		w.Write(testSaltFailure)
	}
	return httptest.NewServer(http.HandlerFunc(handler))
}

// TestVagrantCloud tests against a mock Salt REST API to validate its cloud provider features
func TestVagrantCloud(t *testing.T) {
	server := startSaltTestServer()
	defer server.Close()

	vagrantCloud := &VagrantCloud{
		saltURL:  server.URL,
		saltUser: "vagrant",
		saltPass: "vagrant",
		saltAuth: "pam",
	}

	instances, err := vagrantCloud.List("")
	if err != nil {
		t.Fatalf("There was an error listing instances %s", err)
	}

	if len(instances) != 1 {
		t.Fatalf("Incorrect number of instances returned")
	}

	// no DNS in vagrant cluster, so we return IP as hostname
	expectedInstanceHost := "10.245.2.2"
	expectedInstanceIP := "10.245.2.2"

	if instances[0] != expectedInstanceHost {
		t.Fatalf("Invalid instance returned")
	}

	addrs, err := vagrantCloud.NodeAddresses(instances[0])
	if err != nil {
		t.Fatalf("Unexpected error, should have returned valid NodeAddresses: %s", err)
	}
	if len(addrs) != 1 {
		t.Fatalf("should have returned exactly one NodeAddress: %v", addrs)
	}
	if addrs[0].Address != expectedInstanceIP {
		t.Fatalf("Invalid IP address returned")
	}
}
