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

package flag

import (
	"crypto/tls"
	"fmt"
	"math"
	"strings"
)

// TLSCurvePreferences returns a list of Go's crypto/tls CurveID values from the ids passed.
// The supported values depend on the Go version used.
// See https://pkg.go.dev/crypto/tls#CurveID for values supported for each Go version.
func TLSCurvePreferences(curveIDs []int32) ([]tls.CurveID, error) {
	if len(curveIDs) == 0 {
		return nil, nil
	}
	seen := make(map[int32]bool, len(curveIDs))
	result := make([]tls.CurveID, 0, len(curveIDs))
	for _, id := range curveIDs {
		if id <= 0 || id > math.MaxUint16 {
			return nil, fmt.Errorf("curve preference %d is out of range (must be 1-%d)", id, math.MaxUint16)
		}
		if seen[id] {
			return nil, fmt.Errorf("duplicate curve preference %d", id)
		}
		seen[id] = true
		curve := tls.CurveID(id)
		if strings.HasPrefix(curve.String(), "CurveID(") {
			return nil, fmt.Errorf("curve preference %d is not supported by the current Go version", id)
		}
		result = append(result, curve)
	}
	return result, nil
}
