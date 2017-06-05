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

package componentconfig

import (
	"strings"
	"testing"

	"github.com/spf13/pflag"
)

func TestIPVar(t *testing.T) {
	defaultIP := "0.0.0.0"
	cases := []struct {
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
	for _, c := range cases {
		fs := pflag.NewFlagSet("blah", pflag.PanicOnError)
		ip := defaultIP
		fs.Var(IPVar{&ip}, "ip", "the ip")

		var err error
		func() {
			defer func() {
				if r := recover(); r != nil {
					err = r.(error)
				}
			}()
			fs.Parse(strings.Split(c.argc, " "))
		}()

		if c.expectErr && err == nil {
			t.Errorf("did not observe an expected error")
			continue
		}
		if !c.expectErr && err != nil {
			t.Errorf("observed an unexpected error")
			continue
		}
		if c.expectVal != ip {
			t.Errorf("unexpected ip: expected %q, saw %q", c.expectVal, ip)
		}
	}
}
