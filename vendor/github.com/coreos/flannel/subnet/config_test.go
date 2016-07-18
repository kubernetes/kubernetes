// Copyright 2015 flannel authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package subnet

import (
	"testing"
)

func TestConfigDefaults(t *testing.T) {
	s := `{ "network": "10.3.0.0/16" }`

	cfg, err := ParseConfig(s)
	if err != nil {
		t.Fatalf("ParseConfig failed: %s", err)
	}

	expectedNet := "10.3.0.0/16"
	if cfg.Network.String() != expectedNet {
		t.Errorf("Network mismatch: expected %s, got %s", expectedNet, cfg.Network)
	}

	if cfg.SubnetMin.String() != "10.3.1.0" {
		t.Errorf("SubnetMin mismatch, expected 10.3.1.0, got %s", cfg.SubnetMin)
	}

	if cfg.SubnetMax.String() != "10.3.255.0" {
		t.Errorf("SubnetMax mismatch, expected 10.3.255.0, got %s", cfg.SubnetMax)
	}

	if cfg.SubnetLen != 24 {
		t.Errorf("SubnetLen mismatch: expected 24, got %d", cfg.SubnetLen)
	}
}

func TestConfigOverrides(t *testing.T) {
	s := `{ "Network": "10.3.0.0/16", "SubnetMin": "10.3.5.0", "SubnetMax": "10.3.8.0", "SubnetLen": 28 }`

	cfg, err := ParseConfig(s)
	if err != nil {
		t.Fatalf("ParseConfig failed: %s", err)
	}

	expectedNet := "10.3.0.0/16"
	if cfg.Network.String() != expectedNet {
		t.Errorf("Network mismatch: expected %s, got %s", expectedNet, cfg.Network)
	}

	if cfg.SubnetMin.String() != "10.3.5.0" {
		t.Errorf("SubnetMin mismatch: expected 10.3.5.0, got %s", cfg.SubnetMin)
	}

	if cfg.SubnetMax.String() != "10.3.8.0" {
		t.Errorf("SubnetMax mismatch: expected 10.3.8.0, got %s", cfg.SubnetMax)
	}

	if cfg.SubnetLen != 28 {
		t.Errorf("SubnetLen mismatch: expected 28, got %d", cfg.SubnetLen)
	}
}
