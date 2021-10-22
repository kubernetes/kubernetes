// Copyright 2018 Microsoft Corporation
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

package dirs

import "testing"

func TestGetSubdirs(t *testing.T) {
	sd, err := GetSubdirs("../..")
	if err != nil {
		t.Fatalf("failed to get subdirs: %v", err)
	}
	if len(sd) == 0 {
		t.Fatal("unexpected zero length subdirs")
	}
}

func TestGetSubdirsEmpty(t *testing.T) {
	sd, err := GetSubdirs(".")
	if err != nil {
		t.Fatalf("failed to get subdirs: %v", err)
	}
	if len(sd) != 0 {
		t.Fatal("expected zero length subdirs")
	}
}

func TestGetSubdirsNoExist(t *testing.T) {
	sd, err := GetSubdirs("../thisdoesntexist")
	if err == nil {
		t.Fatal("expected nil error")
	}
	if sd != nil {
		t.Fatal("expected nil subdirs")
	}
}
