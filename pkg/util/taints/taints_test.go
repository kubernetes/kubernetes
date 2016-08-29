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

package taints

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"

	"github.com/spf13/pflag"
)

func TestTaintsVar(t *testing.T) {
	cases := []struct {
		f   string
		err bool
		t   []api.Taint
	}{
		{
			f: "",
			t: []api.Taint(nil),
		},
		{
			f: "--t=foo=bar:NoSchedule",
			t: []api.Taint{{Key: "foo", Value: "bar", Effect: "NoSchedule"}},
		},
		{
			f: "--t=foo=bar:NoSchedule,bing=bang:PreferNoSchedule",
			t: []api.Taint{
				{Key: "foo", Value: "bar", Effect: api.TaintEffectNoSchedule},
				{Key: "bing", Value: "bang", Effect: api.TaintEffectPreferNoSchedule},
			},
		},
	}

	for i, c := range cases {
		args := append([]string{"test"}, strings.Fields(c.f)...)
		cli := pflag.NewFlagSet("test", pflag.ContinueOnError)
		var taints []api.Taint
		cli.Var(NewTaintsVar(&taints), "t", "bar")

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
