// Copyright 2015 Google Inc. All Rights Reserved.
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

package summary

import (
	"reflect"
	"testing"

	info "github.com/google/cadvisor/info/v2"
)

func createSample(i uint64) info.Usage {
	usage := info.Usage{}
	usage.PercentComplete = 100
	usage.Cpu = info.Percentiles{
		Present: true,
		Mean:    i * 50,
		Max:     i * 100,
		Ninety:  i * 90,
	}
	usage.Memory = info.Percentiles{
		Present: true,
		Mean:    i * 50 * 1024,
		Max:     i * 100 * 1024,
		Ninety:  i * 90 * 1024,
	}
	return usage
}

func expectSize(t *testing.T, b *SamplesBuffer, expectedSize int) {
	if b.Size() != expectedSize {
		t.Errorf("Expected size %d, got %d", expectedSize, b.Size())
	}
}

func expectElements(t *testing.T, b *SamplesBuffer, expected []info.Usage) {

	out := b.RecentStats(b.Size())
	if len(out) != len(expected) {
		t.Errorf("Expected %d elements, got %d", len(expected), len(out))
	}
	for i, el := range out {
		if !reflect.DeepEqual(*el, expected[i]) {
			t.Errorf("Expected elements %v, got %v", expected[i], *el)
		}
	}
}

func TestEmpty(t *testing.T) {
	b := NewSamplesBuffer(5)
	expectSize(t, b, 0)
	expectElements(t, b, []info.Usage{})
}

func TestAddSingleSample(t *testing.T) {
	b := NewSamplesBuffer(5)

	sample := createSample(1)
	b.Add(sample)
	expectSize(t, b, 1)
	expectElements(t, b, []info.Usage{sample})
}

func TestFullBuffer(t *testing.T) {
	maxSize := 5
	b := NewSamplesBuffer(maxSize)
	samples := []info.Usage{}
	for i := 0; i < maxSize; i++ {
		sample := createSample(uint64(i))
		samples = append(samples, sample)
		b.Add(sample)
	}
	expectSize(t, b, maxSize)
	expectElements(t, b, samples)
}

func TestOverflow(t *testing.T) {
	maxSize := 5
	overflow := 2
	b := NewSamplesBuffer(maxSize)
	samples := []info.Usage{}
	for i := 0; i < maxSize+overflow; i++ {
		sample := createSample(uint64(i))
		if i >= overflow {
			samples = append(samples, sample)
		}
		b.Add(sample)
	}
	expectSize(t, b, maxSize)
	expectElements(t, b, samples)
}

func TestReplaceAll(t *testing.T) {
	maxSize := 5
	b := NewSamplesBuffer(maxSize)
	samples := []info.Usage{}
	for i := 0; i < maxSize*2; i++ {
		sample := createSample(uint64(i))
		if i >= maxSize {
			samples = append(samples, sample)
		}
		b.Add(sample)
	}
	expectSize(t, b, maxSize)
	expectElements(t, b, samples)
}
