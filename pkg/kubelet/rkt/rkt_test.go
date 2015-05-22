/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package rkt

import (
	"io/ioutil"
	"os"
	"testing"
)

const (
	file = `version: "1.0"
trustedKeys:
- name: coreos.com/etcd
  prefix: coreos.com/etcd
  root: false
  data: LS0tLS1CRUdJTiBQR1AgUFVCTElDIEtFWSBCTE9DSy0tLS0tClZlcnNpb246IEdudVBHIHYxCgptUUlOQkZUQ25NUUJFQUMvNDliR2JTdENwYTNwZWVqKy80Mm1vYmZ1R2JUY2RtY0dHd1labWlnUDBLbDBUUFpLCnpjSXhoVlpPcjNJVFl1QXg4VDFXV0pWYjcvci95NHJvVW9nRHJVU1RtMm5BYkxQOHhwOFFuL04xemFYRnlFdEoKV1RhTHVQSTJtcTY0M3g4ZzdmQWlKWTE5SlJqRmJZVlZaTHI1T01PdkhyT2R0WVZOMzFBUlplU3htcVA1eUZOVwo5RGdvWEcwL3c4MEVPWElzV1dvSmdqYUtMeWUxNUxISTg2TWpQdGhYeVQ1SzIxMmVmeFBmZlNrMWhjYTA0RG1rCkk1dkNNSEMxUTJEYnJsaWxoUzBEVGYrbFNLMllna2FIUFdpTlNaYjNYdmp3YlU4cVFNenlmblFjUXJRbGZtNC8KMWZITGoyYld5QVN6TkcvTU9KQ1EySnlFeUl6YlMyTTRqS2NmRnhhS0t1SkE1UHdkZmpiUlRrdnhBS2JGY2RjNQpFUjdEM1FvRU94Z1JETXBwSGFpaEtOSS9UNGRQSXVxeVVjenEzaWE5ZkdmclFGUk9BSXhkQW5CcXp6QlZhZXR2CkZZRlZqSmxBaEdzV1dFaHVPM1A3cUd3d1I3Q3RQa1d2dnNNVDhDWWRIUDJoN3VPck9aaW9HQ1ErMDlZQkdwSjkKTHp3Q0tIaVYyczQvYUJWZkxoanR0R3FYRytQVy9Lemc1cnR3QWpTZGVvb1RoS1FMUWsvb2sxVHRGeWRnTlZOSAprU1BOZGhnaVRXbE5tSzhRajNDMXpxWm1jUHp2K2M2eTZmNzlHVEwwK0h6Nmdxbk1HZElwRkp0bWJ1c1U3OS8yCk1rRHF1bUJBc2x2d203aDg1czBjY0t3WkNHMVZlbHloR0xhd1Z5eGluMFVoTFdsellkNlNMNUl1dVFBUkFRQUIKdENkRGIzSmxUMU1nUVVOSklFSjFhV3hrWlhJZ1BISmxiR1ZoYzJWQVkyOXlaVzl6TG1OdmJUNkpBamdFRXdFQwpBQ0lGQWxUQ25NUUNHd01HQ3drSUJ3TUNCaFVJQWdrS0N3UVdBZ01CQWg0QkFoZUFBQW9KRUZJUXZZaUlHQ0dRCm54WVArd2NWQ2xYRDF0NW9UN213dllaUGZGOS8raXRPU0h2TjYrK1Q0Q0NGa1JWZElONy9HL091MXdSYi9mVisKUDI3UmM3Z25LK2piUUpxVWE4YUVzTlNXWlQvMUE4VmFRNTFvclFkVjgwWlJPenJKUExCQjB3NGZrRXNTRVNPKwpVdXo5WnNpRU9oWmYwQVRrYWZyRjFqZmVwR1htb0hKeExKK2JLUytLbEFFaHRjZUIxamlXQWFmR1B5OTlYVUFmCk1CSjkwNUY2Ylh2RHFRb3YrOVUzdnlVR253QTZ5bUNDcUtJb0NWeDNHZEtPaDVVYXFDOHBIYUY3b0k3eGt2NWIKNVdNYWp6dVh3S0J5NktvZHVIVG5XN1k1ZzFhb0d3VzVGREZvRXE0THNCdmN4ZVVJNk9NVkVXQ1ZlNTAyRSs0UwpsV3gyZ0V2RmEzM3d5NzdrWXAyWnZIVG9ZNXRTamlJaThRb2NSMElnZkxxRTNQMVpNUGU5WVhrNUV2ZUVaSDZRClZ0UTh6MWt0dVpDVlFxcnRURWVlcm5FZFNGc1RWRlNvV1VzTkpWMUZNbGdpc1paMGxqb0ZKckg3LzdHa1loSzkKRFQ3T2NyWm55WkRVa0VJaVZhcVd3ald3NUluZzRJSEV4QXErUGxYRHdyQTBRY0gyVVc5SXVyREN2UGlYSHB4aQpEMFYyMW9FYkFOVWRHdFljT1NEUkJiWmxzb25JTkpaUTFBZDJYclk4a2ZaMnNaWEFadUJaYkgwK2RFeDZ6bXVhCkMwSUdOM1NMRnNlU2NxSlo1RzFqb1lZcU9LT1V3ZUVya3pBLzYyS2FqMzFTVm9RRHBaeU1xdHdUampaYUZUOE4KZk1rQnRhTTNrbmFGb25IWmMxOUJEMUZPaXNoUlRoQ0NxMlR5OEhVb04yRms3dzBsCj1iWWw3Ci0tLS0tRU5EIFBHUCBQVUJMSUMgS0VZIEJMT0NLLS0tLS0=`
)

func TestReadConfigFile(t *testing.T) {
	if os.Getenv("TEST_RKT") == "" {
		return
	}
	f, err := ioutil.TempFile("", "")
	if err != nil {
		t.Fatalf("Failed to create tmp file: %v", err)
	}
	if _, err := f.WriteString(file); err != nil {
		t.Fatalf("Failed to write tmp file: %v", err)
	}
	f.Close()

	if err := readConfigFile(f.Name()); err != nil {
		t.Errorf("Failed to read config file: %v", err)
	}
}
