/*
Copyright 2023 The Kubernetes Authors.

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

package unittests_test

import (
	"reflect"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"k8s.io/kubernetes/test/e2e/framework"
)

func TestNewFrameworkWithCustomTimeouts(t *testing.T) {
	defaultF := framework.NewDefaultFramework("test")
	customTimeouts := &framework.TimeoutContext{
		PodStart:  5 * time.Second,
		PodDelete: time.Second,
	}
	customF := framework.NewFrameworkWithCustomTimeouts("test", customTimeouts)

	defaultF.Timeouts.PodStart = customTimeouts.PodStart
	defaultF.Timeouts.PodDelete = customTimeouts.PodDelete
	assert.Equal(t, customF.Timeouts, defaultF.Timeouts)
}

func TestNewFramework(t *testing.T) {
	f := framework.NewDefaultFramework("test")

	timeouts := reflect.ValueOf(f.Timeouts).Elem()
	for i := 0; i < timeouts.NumField(); i++ {
		value := timeouts.Field(i)
		if value.IsZero() {
			t.Errorf("%s in Framework.Timeouts was not set.", reflect.TypeOf(*f.Timeouts).Field(i).Name)
		}
	}
}
