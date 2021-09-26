// Copyright 2017 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"strings"
	"testing"
)

func TestBuddyInfo(t *testing.T) {
	buddyInfo, err := getProcFixtures(t).BuddyInfo()
	if err != nil {
		t.Fatal(err)
	}

	if want, got := "DMA", buddyInfo[0].Zone; want != got {
		t.Errorf("want Node 0, Zone %s, got %s", want, got)
	}

	if want, got := "Normal", buddyInfo[2].Zone; want != got {
		t.Errorf("want Node 0, Zone %s, got %s", want, got)
	}

	if want, got := 4381.0, buddyInfo[2].Sizes[0]; want != got {
		t.Errorf("want Node 0, Zone Normal %f, got %f", want, got)
	}

	if want, got := 572.0, buddyInfo[1].Sizes[1]; want != got {
		t.Errorf("want Node 0, Zone DMA32 %f, got %f", want, got)
	}
}

func TestParseBuddyInfoShort(t *testing.T) {

	testdata := `Node 0, zone
Node 0, zone
Node 0, zone
`
	reader := strings.NewReader(testdata)
	_, err := parseBuddyInfo(reader)
	if err == nil {
		t.Fatalf("expected error, but none occurred")
	}
	if want, got := "invalid number of fields when parsing buddyinfo", err.Error(); want != got {
		t.Fatalf("wrong error returned, wanted %q, got %q", want, got)
	}
}

func TestParseBuddyInfoSizeMismatch(t *testing.T) {

	testdata := `Node 0, zone      DMA      1      0      1      0      2      1      1      0      1      1      3
Node 0, zone    DMA32    759    572    791    475    194     45     12      0      0      0      0      0
Node 0, zone   Normal   4381   1093    185   1530    567    102      4      0      0      0
`
	reader := strings.NewReader(testdata)
	_, err := parseBuddyInfo(reader)
	if err == nil {
		t.Fatalf("expected error, but none occurred")
	}
	if want, got := "mismatch in number of buddyinfo buckets", err.Error(); !strings.HasPrefix(got, want) {
		t.Fatalf("wrong error returned, wanted prefix %q, got %q", want, got)
	}
}
