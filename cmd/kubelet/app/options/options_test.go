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

package options

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/util/diff"
	utilflag "k8s.io/apiserver/pkg/util/flag"
)

func newKubeletServerOrDie() *KubeletServer {
	s, err := NewKubeletServer()
	if err != nil {
		panic(err)
	}
	return s
}

func cleanFlags(s *KubeletServer) {
	s.DynamicConfigDir = utilflag.NewStringFlag(s.DynamicConfigDir.Value())
}

// TestRoundTrip ensures that flag values from the Kubelet can be serialized
// to arguments and then read back and have the same value. Also catches cases
// where the default value reported by the flag is not actually allowed to be
// specified.
func TestRoundTrip(t *testing.T) {
	testCases := []struct {
		name          string
		inputFlags    func() *KubeletServer
		outputFlags   func() *KubeletServer
		flagDefaulter func(*pflag.FlagSet)
		skipDefault   bool
		err           bool
		expectArgs    bool
	}{
		{
			name:          "default flags are eliminated",
			inputFlags:    newKubeletServerOrDie,
			outputFlags:   newKubeletServerOrDie,
			flagDefaulter: newKubeletServerOrDie().AddFlags,
			err:           false,
			expectArgs:    false,
		},
		{
			name:          "default flag values round trip",
			inputFlags:    newKubeletServerOrDie,
			outputFlags:   newKubeletServerOrDie,
			flagDefaulter: func(*pflag.FlagSet) {},
			err:           false,
			expectArgs:    true,
		},
		{
			name: "nil address does not fail for optional argument",
			inputFlags: func() *KubeletServer {
				s := newKubeletServerOrDie()
				s.HealthzBindAddress = ""
				return s
			},
			outputFlags: func() *KubeletServer {
				s := newKubeletServerOrDie()
				s.HealthzBindAddress = ""
				return s
			},
			flagDefaulter: func(*pflag.FlagSet) {},
			err:           false,
			expectArgs:    true,
		},
	}
	for _, testCase := range testCases {
		modifiedFlags := testCase.inputFlags()
		args := asArgs(modifiedFlags.AddFlags, testCase.flagDefaulter)
		if testCase.expectArgs != (len(args) > 0) {
			t.Errorf("%s: unexpected args: %v", testCase.name, args)
			continue
		}
		t.Logf("%s: args: %v", testCase.name, args)
		flagSet := pflag.NewFlagSet("output", pflag.ContinueOnError)
		outputFlags := testCase.outputFlags()
		outputFlags.AddFlags(flagSet)
		if err := flagSet.Parse(args); err != nil {
			if !testCase.err {
				t.Errorf("%s: unexpected flag error: %v", testCase.name, err)
			}
			continue
		}
		cleanFlags(outputFlags)
		if !reflect.DeepEqual(modifiedFlags, outputFlags) {
			t.Errorf("%s: flags did not round trip: %s", testCase.name, diff.ObjectReflectDiff(modifiedFlags, outputFlags))
			continue
		}
	}
}

func asArgs(fn, defaultFn func(*pflag.FlagSet)) []string {
	fs := pflag.NewFlagSet("extended", pflag.ContinueOnError)
	fn(fs)
	defaults := pflag.NewFlagSet("defaults", pflag.ContinueOnError)
	defaultFn(defaults)
	var args []string
	fs.VisitAll(func(flag *pflag.Flag) {
		// if the flag implements utilflag.OmitEmpty and the value is Empty, then just omit it from the command line
		if omit, ok := flag.Value.(utilflag.OmitEmpty); ok && omit.Empty() {
			return
		}
		s := flag.Value.String()
		// if the flag has the same value as the default, we can omit it without changing the meaning of the command line
		var defaultValue string
		if defaultFlag := defaults.Lookup(flag.Name); defaultFlag != nil {
			defaultValue = defaultFlag.Value.String()
			if s == defaultValue {
				return
			}
		}
		// if the flag is a string slice, each element is specified with an independent flag invocation
		if values, err := fs.GetStringSlice(flag.Name); err == nil {
			for _, s := range values {
				args = append(args, fmt.Sprintf("--%s=%s", flag.Name, s))
			}
		} else {
			if len(s) == 0 {
				s = defaultValue
			}
			args = append(args, fmt.Sprintf("--%s=%s", flag.Name, s))
		}
	})
	return args
}
