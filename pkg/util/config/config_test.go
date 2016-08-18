/*
Copyright 2014 The Kubernetes Authors.

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

package config

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/spf13/pflag"
)

func TestConfigurationChannels(t *testing.T) {
	mux := NewMux(nil)
	channelOne := mux.Channel("one")
	if channelOne != mux.Channel("one") {
		t.Error("Didn't get the same muxuration channel back with the same name")
	}
	channelTwo := mux.Channel("two")
	if channelOne == channelTwo {
		t.Error("Got back the same muxuration channel for different names")
	}
}

type MergeMock struct {
	source string
	update interface{}
	t      *testing.T
}

func (m MergeMock) Merge(source string, update interface{}) error {
	if m.source != source {
		m.t.Errorf("Expected %s, Got %s", m.source, source)
	}
	if !reflect.DeepEqual(m.update, update) {
		m.t.Errorf("Expected %s, Got %s", m.update, update)
	}
	return nil
}

func TestMergeInvoked(t *testing.T) {
	merger := MergeMock{"one", "test", t}
	mux := NewMux(&merger)
	mux.Channel("one") <- "test"
}

func TestMergeFuncInvoked(t *testing.T) {
	ch := make(chan bool)
	mux := NewMux(MergeFunc(func(source string, update interface{}) error {
		if source != "one" {
			t.Errorf("Expected %s, Got %s", "one", source)
		}
		if update.(string) != "test" {
			t.Errorf("Expected %s, Got %s", "test", update)
		}
		ch <- true
		return nil
	}))
	mux.Channel("one") <- "test"
	<-ch
}

func TestSimultaneousMerge(t *testing.T) {
	ch := make(chan bool, 2)
	mux := NewMux(MergeFunc(func(source string, update interface{}) error {
		switch source {
		case "one":
			if update.(string) != "test" {
				t.Errorf("Expected %s, Got %s", "test", update)
			}
		case "two":
			if update.(string) != "test2" {
				t.Errorf("Expected %s, Got %s", "test2", update)
			}
		default:
			t.Errorf("Unexpected source, Got %s", update)
		}
		ch <- true
		return nil
	}))
	source := mux.Channel("one")
	source2 := mux.Channel("two")
	source <- "test"
	source2 <- "test2"
	<-ch
	<-ch
}

func TestBroadcaster(t *testing.T) {
	b := NewBroadcaster()
	b.Notify(struct{}{})

	ch := make(chan bool, 2)
	b.Add(ListenerFunc(func(object interface{}) {
		if object != "test" {
			t.Errorf("Expected %s, Got %s", "test", object)
		}
		ch <- true
	}))
	b.Add(ListenerFunc(func(object interface{}) {
		if object != "test" {
			t.Errorf("Expected %s, Got %s", "test", object)
		}
		ch <- true
	}))
	b.Notify("test")
	<-ch
	<-ch
}

func TestFeatureConfigFlag(t *testing.T) {
	tests := []struct {
		arg         string
		enableAlpha bool
		parseError  error
	}{
		{fmt.Sprintf("--%s=enableFooBarBaz=maybeidk", flagName), false, fmt.Errorf("unrecognized key: enableFooBarBaz")},
		{fmt.Sprintf("--%s=", flagName), false, nil},
		{fmt.Sprintf("--%s=enableAllAlpha=false", flagName), false, nil},
		{fmt.Sprintf("--%s=enableAllAlpha=true", flagName), true, nil},
		{fmt.Sprintf("--%s=enableAllAlpha=banana", flagName), false, fmt.Errorf("invalid value of enableAllAlpha")},
	}
	for i, test := range tests {
		fs := pflag.NewFlagSet("testfeaturegateflag", pflag.ContinueOnError)
		f := NewFeatureGate()
		f.AddFlag(fs)

		err := fs.Parse([]string{test.arg})
		if test.parseError != nil {
			if !strings.Contains(err.Error(), test.parseError.Error()) {
				t.Errorf("%d: Parse() Expected %v, Got %v", i, test.parseError, err)
			}
		} else if err != nil {
			t.Errorf("%d: Parse() Expected nil, Got %v", i, err)
		}
		if alpha := f.AlphaEnabled(); alpha != test.enableAlpha {
			t.Errorf("%d: AlphaEnabled() expected %v, Got %v", i, test.enableAlpha, alpha)
		}
	}
}
