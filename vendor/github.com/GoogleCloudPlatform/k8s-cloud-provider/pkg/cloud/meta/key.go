/*
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package meta

import (
	"fmt"
	"regexp"
)

// Key for a GCP resource.
type Key struct {
	Name   string
	Zone   string
	Region string
}

// KeyType is the type of the key.
type KeyType string

const (
	// Zonal key type.
	Zonal = "zonal"
	// Regional key type.
	Regional = "regional"
	// Global key type.
	Global = "global"
)

var (
	// locationRegexp is the format of regions/zone names in GCE.
	locationRegexp = regexp.MustCompile("^[a-z](?:[-a-z0-9]+)?$")
)

// ZonalKey returns the key for a zonal resource.
func ZonalKey(name, zone string) *Key {
	return &Key{name, zone, ""}
}

// RegionalKey returns the key for a regional resource.
func RegionalKey(name, region string) *Key {
	return &Key{name, "", region}
}

// GlobalKey returns the key for a global resource.
func GlobalKey(name string) *Key {
	return &Key{name, "", ""}
}

// Type returns the type of the key.
func (k *Key) Type() KeyType {
	switch {
	case k.Zone != "":
		return Zonal
	case k.Region != "":
		return Regional
	default:
		return Global
	}
}

// String returns a string representation of the key.
func (k Key) String() string {
	switch k.Type() {
	case Zonal:
		return fmt.Sprintf("Key{%q, zone: %q}", k.Name, k.Zone)
	case Regional:
		return fmt.Sprintf("Key{%q, region: %q}", k.Name, k.Region)
	default:
		return fmt.Sprintf("Key{%q}", k.Name)
	}
}

// Valid is true if the key is valid.
func (k *Key) Valid() bool {
	if k.Zone != "" && k.Region != "" {
		return false
	}
	switch {
	case k.Region != "":
		return locationRegexp.Match([]byte(k.Region))
	case k.Zone != "":
		return locationRegexp.Match([]byte(k.Zone))
	}
	return true
}

// KeysToMap creates a map[Key]bool from a list of keys.
func KeysToMap(keys ...Key) map[Key]bool {
	ret := map[Key]bool{}
	for _, k := range keys {
		ret[k] = true
	}
	return ret
}
