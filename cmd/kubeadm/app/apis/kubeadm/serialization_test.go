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

package kubeadm_test

import (
	"math/rand"
	"testing"

	apitesting "k8s.io/apimachinery/pkg/api/testing"
	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/install"
	"k8s.io/kubernetes/pkg/api"
	kapitesting "k8s.io/kubernetes/pkg/api/testing"
)

const (
	seed = 1
)

func TestRoundTripTypes(t *testing.T) {

	groupFactoryRegistry := make(announced.APIGroupFactoryRegistry)
	registry := registered.NewOrDie("kubeadmtest")
	scheme := runtime.NewScheme()

	install.Install(groupFactoryRegistry, registry, scheme)
	fuzzer := apitesting.FuzzerFor(kapitesting.FuzzerFuncs(t, api.Codecs), rand.NewSource(seed))
	apitesting.RoundTripTypesWithoutProtobuf(t, scheme, api.Codecs, fuzzer, nil)
}
