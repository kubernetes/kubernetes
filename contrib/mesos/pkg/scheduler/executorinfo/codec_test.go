/*
Copyright 2015 The Kubernetes Authors.

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

package executorinfo

import (
	"bytes"
	"reflect"
	"testing"

	"github.com/mesos/mesos-go/mesosproto"
	"github.com/mesos/mesos-go/mesosutil"
)

func TestEncodeDecode(t *testing.T) {
	want := []*mesosproto.Resource{
		scalar("cpus", 0.1, "*"),
		scalar("mem", 64.0, "*"),
		scalar("mem", 128.0, "public_slave"),
	}

	var buf bytes.Buffer
	if err := EncodeResources(&buf, want); err != nil {
		t.Error(err)
	}

	got, err := DecodeResources(&buf)
	if err != nil {
		t.Error(err)
	}

	if ok := reflect.DeepEqual(want, got); !ok {
		t.Errorf("want %v got %v", want, got)
	}
}

func TestEncodeDecodeNil(t *testing.T) {
	var buf bytes.Buffer
	if err := EncodeResources(&buf, nil); err != nil {
		t.Error(err)
	}

	if buf.String() != "" {
		t.Errorf("expected empty string but got %q", buf.String())
	}

	if _, err := DecodeResources(&buf); err == nil {
		t.Errorf("expected error but got none")
	}
}

func scalar(name string, value float64, role string) *mesosproto.Resource {
	res := mesosutil.NewScalarResource(name, value)
	res.Role = &role
	return res
}
