/*
Copyright 2019 The Kubernetes Authors.

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

package renewal

import (
	"crypto/x509"
	"math"
	"testing"
	"time"
)

func TestExpirationInfo(t *testing.T) {
	validity := 365 * 24 * time.Hour
	cert := &x509.Certificate{
		NotAfter: time.Now().Add(validity),
	}

	e := newExpirationInfo("x", cert, false)

	if math.Abs(float64(validity-e.ResidualTime())) > float64(5*time.Second) { // using 5s of tolerance because the function is not deterministic (it uses time.Now()) and we want to avoid flakes
		t.Errorf("expected IsInRenewalWindow equal to %v, saw %v", validity, e.ResidualTime())
	}
}
