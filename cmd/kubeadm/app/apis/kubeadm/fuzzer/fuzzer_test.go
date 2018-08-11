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

package fuzzer

// TODO: Fuzzing rouudtrip tests are currently disabled in the v1.12 cycle due to the
// v1alpha2 -> v1alpha3 migration. As the ComponentConfigs were embedded in the structs
// earlier now have moved out it's not possible to do a lossless roundtrip "the normal way"
// When we support v1alpha3 and higher only, we can reenable this

/*import (
	"testing"

	"k8s.io/apimachinery/pkg/api/apitesting/roundtrip"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
)

func TestRoundTripTypes(t *testing.T) {
	roundtrip.RoundTripTestForAPIGroup(t, scheme.AddToScheme, Funcs)
}*/
