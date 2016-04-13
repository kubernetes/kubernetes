// Copyright 2015 The rkt Authors
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

package main

import (
	"reflect"
	"testing"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

func setupPodStoreTest(t *testing.T) (*podStore, *types.UUID, string, string) {
	ps := newPodStore()

	uuid, err := types.NewUUID("de305d54-75b4-431b-adb2-eb6b9e546013")
	if err != nil {
		panic("bad uuid literal")
	}

	ip := "1.2.3.4"
	app := "myapp"

	pm := &schema.PodManifest{}
	ps.addPod(uuid, ip, pm)

	im := &schema.ImageManifest{}
	err = ps.addApp(uuid, app, im)
	if err != nil {
		t.Fatalf("addApp failed with %v", err)
	}

	return ps, uuid, ip, app
}

func TestPodStoreAddApp(t *testing.T) {
	ps, _, _, app := setupPodStoreTest(t)

	uuid2, err := types.NewUUID("fe305d54-75b4-431b-adb2-eb6b9e546013")
	if err != nil {
		panic("bad uuid literal")
	}

	im := &schema.ImageManifest{}
	if err = ps.addApp(uuid2, app, im); err != errPodNotFound {
		t.Errorf("addApp with unknown pod returned: %v", err)
	}
}

func TestPodStoreGetUUID(t *testing.T) {
	ps, uuid, ip, _ := setupPodStoreTest(t)

	u, err := ps.getUUID(ip)
	if err != nil {
		t.Errorf("getUUID failed with %v", err)
	}

	if !reflect.DeepEqual(*u, *uuid) {
		t.Errorf("getUUID mismatch: got %v, expected %v", u, uuid)
	}

	if _, err := ps.getUUID("2.3.4.5"); err != errPodNotFound {
		t.Errorf("getUUID with unknown pod returned: %v", err)
	}
}

func TestPodStoreGetPodManifest(t *testing.T) {
	ps, _, ip, _ := setupPodStoreTest(t)

	if _, err := ps.getPodManifest(ip); err != nil {
		t.Errorf("getPodManifest failed with %v", err)
	}

	if _, err := ps.getPodManifest("2.3.4.5"); err != errPodNotFound {
		t.Errorf("getPodManifest with unknown pood returned %v", err)
	}
}

func TestPodStoreGetManifests(t *testing.T) {
	ps, _, ip, app := setupPodStoreTest(t)

	if _, _, err := ps.getManifests(ip, app); err != nil {
		t.Errorf("getManifests failed with %v", err)
	}

	if _, _, err := ps.getManifests("2.3.4.5", app); err != errPodNotFound {
		t.Errorf("getManifests with unknown pod returned %v", err)
	}

	if _, _, err := ps.getManifests(ip, "foo"); err != errAppNotFound {
		t.Errorf("getManifests with unknown app returned %v", err)
	}
}

func TestPodStoreRemove(t *testing.T) {
	ps, uuid, _, _ := setupPodStoreTest(t)

	err := ps.remove(uuid)
	if err != nil {
		t.Errorf("remove failed with %v", err)
	}

	if err := ps.remove(uuid); err != errPodNotFound {
		t.Errorf("remove with unknown pod returned %v", err)
	}
}
