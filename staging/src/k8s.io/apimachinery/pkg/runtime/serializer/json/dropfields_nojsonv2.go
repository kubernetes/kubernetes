//go:build !goexperiment.jsonv2 && !go1.27

/*
Copyright The Kubernetes Authors.

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

package json

import (
	"encoding/json"
	"io"

	"k8s.io/apimachinery/pkg/runtime"
)

// encodeDroppingFields is the fallback for toolchains without json/v2 (Go < 1.27):
// dropping fields at marshal time requires json/v2's custom marshalers, so here it
// encodes the full object. Requesting drop= on such a build returns the full object,
// which is the safe version-skew default.
func (s *Serializer) encodeDroppingFields(obj runtime.Object, w io.Writer) error {
	return json.NewEncoder(w).Encode(obj)
}
