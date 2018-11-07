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

package ipvs

import (
	"fmt"
	"testing"

	utilsexec "k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

func TestRequiredIPVSKernelModulesAvailableCheck(t *testing.T) {
	cases := []struct {
		caseName string

		loadedKernel  string
		kernelVersion string
		builtinKernel string
		modinfoError  error

		expectErrors   bool
		expectWarnings bool
	}{
		{
			caseName:       "no loaded, no installed and no builtin kernel modules",
			loadedKernel:   "",
			kernelVersion:  "3.13.0-24-generic",
			builtinKernel:  "",
			modinfoError:   fmt.Errorf("modinfo error"),
			expectErrors:   false,
			expectWarnings: true,
		},
		{
			caseName:      "no loaded, no installed and missing builtin kernel modules",
			loadedKernel:  "",
			kernelVersion: "3.13.0-24-generic",
			builtinKernel: "kernel/net/netfilter/ipvs/ip_vs.ko\n" +
				"kernel/net/ipv4/netfilter/nf_conntrack_ipv4.ko",
			modinfoError:   fmt.Errorf("modinfo error"),
			expectErrors:   false,
			expectWarnings: true,
		},
		{
			caseName:      "no loaded, some installed and some builtin kernel modules",
			loadedKernel:  "",
			kernelVersion: "3.13.0-24-generic",
			builtinKernel: "kernel/net/netfilter/ipvs/ip_vs.ko\n" +
				"kernel/net/netfilter/ipvs/ip_vs_rr.ko\n" +
				"kernel/net/netfilter/ipvs/ip_vs_wrr.ko\n",
			modinfoError:   nil,
			expectErrors:   false,
			expectWarnings: false,
		},
		{
			caseName:      "no loaded, no installed and all builtin kernel modules",
			loadedKernel:  "",
			kernelVersion: "3.13.0-24-generic",
			builtinKernel: "kernel/net/netfilter/ipvs/ip_vs.ko\n" +
				"kernel/net/netfilter/ipvs/ip_vs_rr.ko\n" +
				"kernel/net/netfilter/ipvs/ip_vs_wrr.ko\n" +
				"kernel/net/netfilter/ipvs/ip_vs_sh.ko\n" +
				"kernel/net/ipv4/netfilter/nf_conntrack_ipv4.ko",
			modinfoError:   fmt.Errorf("modinfo error"),
			expectErrors:   false,
			expectWarnings: true,
		},
		{
			caseName: "all loaded, no installed and no builtin kernel modules",
			loadedKernel: "ip_vs\n" + "ip_vs_wrr\n" + "nf_conntrack_ipv4\n" +
				"ip_vs_rr\n" + "ip_vs_sh",
			kernelVersion:  "3.13.0-24-generic",
			builtinKernel:  "",
			modinfoError:   fmt.Errorf("modinfo error"),
			expectErrors:   false,
			expectWarnings: false,
		},
		{
			caseName:      "all loaded, all installed and all builtin kernel modules",
			loadedKernel:  "ip_vs\n" + "ip_vs_wrr\n" + "nf_conntrack_ipv4\n" + "ip_vs_rr\n" + "ip_vs_sh",
			kernelVersion: "3.13.0-24-generic",
			builtinKernel: "kernel/net/netfilter/ipvs/ip_vs.ko\n" +
				"kernel/net/netfilter/ipvs/ip_vs_rr.ko\n" +
				"kernel/net/netfilter/ipvs/ip_vs_wrr.ko\n" +
				"kernel/net/netfilter/ipvs/ip_vs_sh.ko\n" +
				"kernel/net/ipv4/netfilter/nf_conntrack_ipv4.ko",
			modinfoError:   nil,
			expectErrors:   false,
			expectWarnings: false,
		},
	}

	for i, tc := range cases {
		fcmd := fakeexec.FakeCmd{
			CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
				func() ([]byte, error) { return []byte(cases[i].kernelVersion), nil },
				func() ([]byte, error) { return []byte(cases[i].loadedKernel), nil },
				func() ([]byte, error) { return []byte("modinfo output"), tc.modinfoError },
				func() ([]byte, error) { return []byte("modinfo output"), tc.modinfoError },
				func() ([]byte, error) { return []byte("modinfo output"), tc.modinfoError },
				func() ([]byte, error) { return []byte("modinfo output"), tc.modinfoError },
				func() ([]byte, error) { return []byte("modinfo output"), tc.modinfoError },
				func() ([]byte, error) { return []byte(cases[i].builtinKernel), nil },
			},
		}

		fexec := fakeexec.FakeExec{
			CommandScript: []fakeexec.FakeCommandAction{
				func(cmd string, args ...string) utilsexec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) utilsexec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) utilsexec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) utilsexec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) utilsexec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) utilsexec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) utilsexec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) utilsexec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			},
		}

		check := RequiredIPVSKernelModulesAvailableCheck{
			Executor: &fexec,
		}
		warnings, errors := check.Check()

		switch {
		case warnings != nil && !tc.expectWarnings:
			t.Errorf("RequiredIPVSKernelModulesAvailableCheck: unexpected warnings for loaded kernel modules %v and builtin kernel modules %v. Warnings: %v", tc.loadedKernel, tc.builtinKernel, warnings)
		case warnings == nil && tc.expectWarnings:
			t.Errorf("RequiredIPVSKernelModulesAvailableCheck: expected warnings for loaded kernel modules %v and builtin kernel modules %v but got nothing", tc.loadedKernel, tc.builtinKernel)
		case errors != nil && !tc.expectErrors:
			t.Errorf("RequiredIPVSKernelModulesAvailableCheck: unexpected errors for loaded kernel modules %v and builtin kernel modules %v. errors: %v", tc.loadedKernel, tc.builtinKernel, errors)
		case errors == nil && tc.expectErrors:
			t.Errorf("RequiredIPVSKernelModulesAvailableCheck: expected errors for loaded kernel modules %v and builtin kernel modules %v but got nothing", tc.loadedKernel, tc.builtinKernel)
		}
	}
}
