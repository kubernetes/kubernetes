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

package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/stretchr/testify/assert"
	"k8s.io/apiserver/pkg/metrics/fake"
)

func Test_multipleCollector_Register(t *testing.T) {
	assert := assert.New(t)
	one := fake.New("one")
	two := fake.New("two")

	registry := &fake.FakeRegistry{}

	m := NewStore(registry, NewGroup(one), NewGroup(two))
	// calling Register() twice to be sure that we register
	// metrics only once.
	m.Register()
	m.Register()

	registredMetrics := registry.GetRegistredMetrics()
	assert.Len(registredMetrics, 2)
	assert.Contains(registredMetrics, one)
	assert.Contains(registredMetrics, two)
}

func Test_multipleCollector_Reset(t *testing.T) {
	assert := assert.New(t)
	one := fake.New("one")
	two := fake.New("two")
	registry := &fake.FakeRegistry{}
	m := NewStore(registry, NewGroup(one), NewGroup(two))

	m.Reset()

	assert.Equal(1, one.ResetCalledCount())
	assert.Equal(1, two.ResetCalledCount())

	m.Reset()
	assert.Equal(2, one.ResetCalledCount())
	assert.Equal(2, two.ResetCalledCount())
}

func TestNew_Default_Prometheus_Registry(t *testing.T) {
	assert := assert.New(t)
	one := fake.New("one")
	m := NewStore(nil, NewGroup(one))

	m.Register()
	assert.True(prometheus.DefaultRegisterer.Unregister(one))

}
