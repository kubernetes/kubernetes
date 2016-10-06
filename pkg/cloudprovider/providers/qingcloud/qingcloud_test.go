/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package qingcloud

import (
	"strings"
	"testing"
)

func TestReadConfig(t *testing.T) {
	_, err := readConfig(nil)
	if err == nil {
		t.Errorf("Should fail when no config is provided: %s", err)
	}

	cfg, err := readConfig(strings.NewReader(`
[Global]
accessKeyID = abc
secretAccessKey = abcdefg
zone = ap1
 `))
	if err != nil {
		t.Fatalf("Should succeed when a valid config is provided: %s", err)
	}
	if cfg.Global.AccessKeyID != "abc" {
		t.Errorf("incorrect accessKeyID: %s", cfg.Global.AccessKeyID)
	}
	if cfg.Global.SecretAccessKey != "abcdefg" {
		t.Errorf("incorrect secretAccessKeyr: %s", cfg.Global.SecretAccessKey)
	}
	if cfg.Global.Zone != "ap1" {
		t.Errorf("incorrect zone: %s", cfg.Global.Zone)
	}
}

func TestZones(t *testing.T) {
	qc := Qingcloud{zone: "ap1"}

	z, ok := qc.Zones()
	if !ok {
		t.Fatalf("Zones() returned false")
	}

	zone, err := z.GetZone()
	if err != nil {
		t.Fatalf("GetZone() returned error: %s", err)
	}

	if zone.Region != qc.zone {
		t.Fatalf("GetZone() returned wrong region (%s)", zone)
	}
}
