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

package flag

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/spf13/pflag"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

func TestIPVar(t *testing.T) {
	defaultIP := "0.0.0.0"
	testCases := []struct {
		argc      string
		expectErr bool
		expectVal string
	}{

		{
			argc:      "blah --ip=1.2.3.4",
			expectVal: "1.2.3.4",
		},
		{
			argc:      "blah --ip=1.2.3.4a",
			expectErr: true,
			expectVal: defaultIP,
		},
	}
	for _, tc := range testCases {
		fs := pflag.NewFlagSet("blah", pflag.PanicOnError)
		ip := defaultIP
		fs.Var(&IPVar{&ip}, "ip", "the ip")

		var err error
		func() {
			defer func() {
				if r := recover(); r != nil {
					err = r.(error)
				}
			}()
			fs.Parse(strings.Split(tc.argc, " "))
		}()

		if tc.expectErr && err == nil {
			t.Errorf("did not observe an expected error")
			continue
		}
		if !tc.expectErr && err != nil {
			t.Errorf("observed an unexpected error: %v", err)
			continue
		}
		if tc.expectVal != ip {
			t.Errorf("unexpected ip: expected %q, saw %q", tc.expectVal, ip)
		}
	}
}

func TestIPPortVar(t *testing.T) {
	defaultIPPort := "0.0.0.0:8080"
	testCases := []struct {
		desc      string
		argc      string
		expectErr bool
		expectVal string
	}{

		{
			desc:      "valid ipv4 1",
			argc:      "blah --ipport=0.0.0.0",
			expectVal: "0.0.0.0",
		},
		{
			desc:      "valid ipv4 2",
			argc:      "blah --ipport=127.0.0.1",
			expectVal: "127.0.0.1",
		},

		{
			desc:      "invalid IP",
			argc:      "blah --ipport=invalidip",
			expectErr: true,
			expectVal: defaultIPPort,
		},
		{
			desc:      "valid ipv4 with port",
			argc:      "blah --ipport=0.0.0.0:8080",
			expectVal: "0.0.0.0:8080",
		},
		{
			desc:      "invalid ipv4 with invalid port",
			argc:      "blah --ipport=0.0.0.0:invalidport",
			expectErr: true,
			expectVal: defaultIPPort,
		},
		{
			desc:      "invalid IP with port",
			argc:      "blah --ipport=invalidip:8080",
			expectErr: true,
			expectVal: defaultIPPort,
		},
		{
			desc:      "valid ipv6 1",
			argc:      "blah --ipport=::1",
			expectVal: "::1",
		},
		{
			desc:      "valid ipv6 2",
			argc:      "blah --ipport=::",
			expectVal: "::",
		},
		{
			desc:      "valid ipv6 with port",
			argc:      "blah --ipport=[::1]:8080",
			expectVal: "[::1]:8080",
		},
		{
			desc:      "invalid ipv6 with port without bracket",
			argc:      "blah --ipport=fd00:f00d:600d:f00d:8080",
			expectErr: true,
			expectVal: defaultIPPort,
		},
	}
	for _, tc := range testCases {
		fs := pflag.NewFlagSet("blah", pflag.PanicOnError)
		ipport := defaultIPPort
		fs.Var(&IPPortVar{&ipport}, "ipport", "the ip:port")

		var err error
		func() {
			defer func() {
				if r := recover(); r != nil {
					err = r.(error)
				}
			}()
			fs.Parse(strings.Split(tc.argc, " "))
		}()

		if tc.expectErr && err == nil {
			t.Errorf("%q: Did not observe an expected error", tc.desc)
			continue
		}
		if !tc.expectErr && err != nil {
			t.Errorf("%q: Observed an unexpected error: %v", tc.desc, err)
			continue
		}
		if tc.expectVal != ipport {
			t.Errorf("%q: Unexpected ipport: expected %q, saw %q", tc.desc, tc.expectVal, ipport)
		}
	}
}

func TestReservedMemoryVar(t *testing.T) {
	resourceNameHugepages1Gi := v1.ResourceName(fmt.Sprintf("%s1Gi", v1.ResourceHugePagesPrefix))
	memory1Gi := resource.MustParse("1Gi")
	testCases := []struct {
		desc      string
		argc      string
		expectErr bool
		expectVal []kubeletconfig.MemoryReservation
	}{
		{
			desc: "valid input",
			argc: "blah --reserved-memory=0:memory=1Gi",
			expectVal: []kubeletconfig.MemoryReservation{
				{
					NumaNode: 0,
					Limits: v1.ResourceList{
						v1.ResourceMemory: memory1Gi,
					},
				},
			},
		},
		{
			desc: "valid input with multiple memory types",
			argc: "blah --reserved-memory=0:memory=1Gi,hugepages-1Gi=1Gi",
			expectVal: []kubeletconfig.MemoryReservation{
				{
					NumaNode: 0,
					Limits: v1.ResourceList{
						v1.ResourceMemory:        memory1Gi,
						resourceNameHugepages1Gi: memory1Gi,
					},
				},
			},
		},
		{
			desc: "valid input with multiple reserved-memory arguments",
			argc: "blah --reserved-memory=0:memory=1Gi,hugepages-1Gi=1Gi --reserved-memory=1:memory=1Gi",
			expectVal: []kubeletconfig.MemoryReservation{
				{
					NumaNode: 0,
					Limits: v1.ResourceList{
						v1.ResourceMemory:        memory1Gi,
						resourceNameHugepages1Gi: memory1Gi,
					},
				},
				{
					NumaNode: 1,
					Limits: v1.ResourceList{
						v1.ResourceMemory: memory1Gi,
					},
				},
			},
		},
		{
			desc: "valid input with ';' as separator for multiple reserved-memory arguments",
			argc: "blah --reserved-memory=0:memory=1Gi,hugepages-1Gi=1Gi;1:memory=1Gi",
			expectVal: []kubeletconfig.MemoryReservation{
				{
					NumaNode: 0,
					Limits: v1.ResourceList{
						v1.ResourceMemory:        memory1Gi,
						resourceNameHugepages1Gi: memory1Gi,
					},
				},
				{
					NumaNode: 1,
					Limits: v1.ResourceList{
						v1.ResourceMemory: memory1Gi,
					},
				},
			},
		},
		{
			desc:      "invalid input",
			argc:      "blah --reserved-memory=bad-input",
			expectVal: nil,
			expectErr: true,
		},
		{
			desc:      "invalid input without memory types",
			argc:      "blah --reserved-memory=0:",
			expectVal: nil,
			expectErr: true,
		},
		{
			desc:      "invalid input with non-integer NUMA node",
			argc:      "blah --reserved-memory=a:memory=1Gi",
			expectVal: nil,
			expectErr: true,
		},
		{
			desc:      "invalid input with invalid limit",
			argc:      "blah --reserved-memory=0:memory=",
			expectVal: nil,
			expectErr: true,
		},
		{
			desc:      "invalid input with invalid memory type",
			argc:      "blah --reserved-memory=0:type=1Gi",
			expectVal: nil,
			expectErr: true,
		},
		{
			desc:      "invalid input with invalid quantity",
			argc:      "blah --reserved-memory=0:memory=1Be",
			expectVal: nil,
			expectErr: true,
		},
	}
	for _, tc := range testCases {
		fs := pflag.NewFlagSet("blah", pflag.PanicOnError)

		var reservedMemory []kubeletconfig.MemoryReservation
		fs.Var(&ReservedMemoryVar{Value: &reservedMemory}, "reserved-memory", "--reserved-memory 0:memory=1Gi,hugepages-1M=2Gi")

		var err error
		func() {
			defer func() {
				if r := recover(); r != nil {
					err = r.(error)
				}
			}()
			fs.Parse(strings.Split(tc.argc, " "))
		}()

		if tc.expectErr && err == nil {
			t.Fatalf("%q: Did not observe an expected error", tc.desc)
		}
		if !tc.expectErr && err != nil {
			t.Fatalf("%q: Observed an unexpected error: %v", tc.desc, err)
		}
		if !apiequality.Semantic.DeepEqual(reservedMemory, tc.expectVal) {
			t.Fatalf("%q: Unexpected reserved-error: expected %v, saw %v", tc.desc, tc.expectVal, reservedMemory)
		}
	}
}

func TestTaintsVar(t *testing.T) {
	cases := []struct {
		f   string
		err bool
		t   []v1.Taint
	}{
		{
			f: "",
			t: []v1.Taint(nil),
		},
		{
			f: "--t=foo=bar:NoSchedule",
			t: []v1.Taint{{Key: "foo", Value: "bar", Effect: "NoSchedule"}},
		},
		{
			f: "--t=baz:NoSchedule",
			t: []v1.Taint{{Key: "baz", Value: "", Effect: "NoSchedule"}},
		},
		{
			f: "--t=foo=bar:NoSchedule,baz:NoSchedule,bing=bang:PreferNoSchedule,qux=:NoSchedule",
			t: []v1.Taint{
				{Key: "foo", Value: "bar", Effect: v1.TaintEffectNoSchedule},
				{Key: "baz", Value: "", Effect: "NoSchedule"},
				{Key: "bing", Value: "bang", Effect: v1.TaintEffectPreferNoSchedule},
				{Key: "qux", Value: "", Effect: "NoSchedule"},
			},
		},
		{
			f: "--t=dedicated-for=user1:NoExecute,baz:NoSchedule,foo-bar=:NoSchedule",
			t: []v1.Taint{
				{Key: "dedicated-for", Value: "user1", Effect: "NoExecute"},
				{Key: "baz", Value: "", Effect: "NoSchedule"},
				{Key: "foo-bar", Value: "", Effect: "NoSchedule"},
			},
		},
	}

	for i, c := range cases {
		args := append([]string{"test"}, strings.Fields(c.f)...)
		cli := pflag.NewFlagSet("test", pflag.ContinueOnError)
		var taints []v1.Taint
		cli.Var(RegisterWithTaintsVar{Value: &taints}, "t", "bar")

		err := cli.Parse(args)
		if err == nil && c.err {
			t.Errorf("[%v] expected error", i)
			continue
		}
		if err != nil && !c.err {
			t.Errorf("[%v] unexpected error: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(c.t, taints) {
			t.Errorf("[%v] unexpected taints:\n\texpected:\n\t\t%#v\n\tgot:\n\t\t%#v", i, c.t, taints)
		}
	}

}
