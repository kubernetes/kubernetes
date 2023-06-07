/*
Copyright 2014 The Kubernetes Authors.

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

package validation

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestIsValidIP(t *testing.T) {
	goodValues := []string{
		"::1",
		"2a00:79e0:2:0:f1c3:e797:93c1:df80",
		"::",
		"2001:4860:4860::8888",
		"::fff:1.1.1.1",
		"1.1.1.1",
		"1.1.1.01",
		"255.0.0.1",
		"1.0.0.0",
		"0.0.0.0",
	}
	for _, val := range goodValues {
		if msgs := IsValidIP(val); len(msgs) != 0 {
			t.Errorf("expected true for %q: %v", val, msgs)
		}
	}

	badValues := []string{
		"[2001:db8:0:1]:80",
		"myhost.mydomain",
		"-1.0.0.0",
		"[2001:db8:0:1]",
		"a",
	}
	for _, val := range badValues {
		if msgs := IsValidIP(val); len(msgs) == 0 {
			t.Errorf("expected false for %q", val)
		}
	}
}

func TestIsValidIPv4Address(t *testing.T) {
	goodValues := []string{
		"1.1.1.1",
		"1.1.1.01",
		"255.0.0.1",
		"1.0.0.0",
		"0.0.0.0",
	}
	for _, val := range goodValues {
		if msgs := IsValidIPv4Address(field.NewPath(""), val); len(msgs) != 0 {
			t.Errorf("expected %q to be valid IPv4 address: %v", val, msgs)
		}
	}

	badValues := []string{
		"[2001:db8:0:1]:80",
		"myhost.mydomain",
		"-1.0.0.0",
		"[2001:db8:0:1]",
		"a",
		"2001:4860:4860::8888",
		"::fff:1.1.1.1",
		"::1",
		"2a00:79e0:2:0:f1c3:e797:93c1:df80",
		"::",
	}
	for _, val := range badValues {
		if msgs := IsValidIPv4Address(field.NewPath(""), val); len(msgs) == 0 {
			t.Errorf("expected %q to be invalid IPv4 address", val)
		}
	}
}

func TestIsValidIPv6Address(t *testing.T) {
	goodValues := []string{
		"2001:4860:4860::8888",
		"2a00:79e0:2:0:f1c3:e797:93c1:df80",
		"2001:0db8:85a3:0000:0000:8a2e:0370:7334",
		"::fff:1.1.1.1",
		"::1",
		"::",
	}

	for _, val := range goodValues {
		if msgs := IsValidIPv6Address(field.NewPath(""), val); len(msgs) != 0 {
			t.Errorf("expected %q to be valid IPv6 address: %v", val, msgs)
		}
	}

	badValues := []string{
		"1.1.1.1",
		"1.1.1.01",
		"255.0.0.1",
		"1.0.0.0",
		"0.0.0.0",
		"[2001:db8:0:1]:80",
		"myhost.mydomain",
		"2001:0db8:85a3:0000:0000:8a2e:0370:7334:2001:0db8:85a3:0000:0000:8a2e:0370:7334",
		"-1.0.0.0",
		"[2001:db8:0:1]",
		"a",
	}
	for _, val := range badValues {
		if msgs := IsValidIPv6Address(field.NewPath(""), val); len(msgs) == 0 {
			t.Errorf("expected %q to be invalid IPv6 address", val)
		}
	}
}
