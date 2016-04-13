// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pubsub

import (
	"net/http"
	"strings"
	"testing"
	"time"

	"google.golang.org/cloud"
)

func TestIsSec(t *testing.T) {
	tests := map[time.Duration]bool{
		time.Second:                    true,
		5 * time.Second:                true,
		time.Hour:                      true,
		time.Millisecond:               false,
		time.Second + time.Microsecond: false,
	}
	for dur, expected := range tests {
		if isSec(dur) != expected {
			t.Errorf("%v is more precise than a second", dur)
		}
	}
}

func TestEmptyAckID(t *testing.T) {
	ctx := cloud.NewContext("project-id", &http.Client{})
	id := []string{"test", ""}
	err := Ack(ctx, "sub", id...)

	if err == nil || !strings.Contains(err.Error(), "index 1") {
		t.Errorf("Ack should report an error indicating the id is empty. Got: %v", err)
	}
}
