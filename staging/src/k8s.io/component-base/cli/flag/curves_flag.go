/*
Copyright 2025 The Kubernetes Authors.

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
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
)

// curvePreferences maps IANA TLS Supported Groups names (a.k.a curves) (keys)[1] to Go crypto/tls CurveID constants (values)[2].
// [1] https://www.iana.org/assignments/tls-parameters/tls-parameters.xml#tls-parameters-8
// [2] https://golang.org/pkg/crypto/tls/#CurveID
// The keys MUST be in lowercase for easier case-insensitive matching.
var curvePreferences = map[string]tls.CurveID{
	"secp256r1":      tls.CurveP256,
	"secp384r1":      tls.CurveP384,
	"secp521r1":      tls.CurveP521,
	"x25519":         tls.X25519,
	"x25519mlkem768": tls.X25519MLKEM768,
}

// PreferredTLSCurveNames returns the list of acceptable curve names using IANA TLS Supported Groups names.
func PreferredTLSCurveNames() []string {
	curveKeys := sets.NewString()
	for key := range curvePreferences {
		curveKeys.Insert(key)
	}
	return curveKeys.List()
}

// TLSCurvePossibleValues returns all acceptable values for TLS curve preferences.
func TLSCurvePossibleValues() []string {
	return PreferredTLSCurveNames()
}

// TLSCurvePreferences returns a list of tls.CurveID values from the IANA curve names passed.
// Curve names are matched case-insensitively.
func TLSCurvePreferences(curveNames []string) ([]tls.CurveID, error) {
	if len(curveNames) == 0 {
		return nil, nil
	}
	curveIDs := make([]tls.CurveID, 0, len(curveNames))
	for _, name := range curveNames {
		// Match curve names case-insensitively.
		curveID, ok := curvePreferences[strings.ToLower(name)]
		if !ok {
			return nil, fmt.Errorf("curve preference %q not supported or doesn't exist", name)
		}
		curveIDs = append(curveIDs, curveID)
	}
	return curveIDs, nil
}
