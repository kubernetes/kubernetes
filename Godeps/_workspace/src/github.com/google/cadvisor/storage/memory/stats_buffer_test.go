// Copyright 2014 Google Inc. All Rights Reserved.
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

package memory

import (
	"strconv"
	"strings"
	"testing"
	"time"

	info "github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
)

func createTime(id int) time.Time {
	var zero time.Time
	return zero.Add(time.Duration(id+1) * time.Second)
}

func createStats(id int32) *info.ContainerStats {
	return &info.ContainerStats{
		Timestamp: createTime(int(id)),
		Cpu: info.CpuStats{
			LoadAverage: id,
		},
	}
}

func expectSize(t *testing.T, sb *StatsBuffer, expectedSize int) {
	if sb.Size() != expectedSize {
		t.Errorf("Expected size %v, got %v", expectedSize, sb.Size())
	}
}

func expectFirstN(t *testing.T, sb *StatsBuffer, expected []int32) {
	expectElements(t, sb.FirstN(sb.Size()), expected)
}

func expectElements(t *testing.T, actual []*info.ContainerStats, expected []int32) {
	if len(actual) != len(expected) {
		t.Errorf("Expected elements %v, got %v", expected, actual)
		return
	}
	for i, el := range actual {
		if el.Cpu.LoadAverage != expected[i] {
			actualElements := make([]string, len(actual))
			for i, element := range actual {
				actualElements[i] = strconv.Itoa(int(element.Cpu.LoadAverage))
			}
			t.Errorf("Expected elements %v, got %v", expected, strings.Join(actualElements, ","))
			return
		}
	}
}

func expectElement(t *testing.T, stat *info.ContainerStats, expected int32) {
	if stat.Cpu.LoadAverage != expected {
		t.Errorf("Expected %d, but received %d", expected, stat.Cpu.LoadAverage)
	}
}

func TestAddAndFirstN(t *testing.T) {
	sb := NewStatsBuffer(5)

	// Add 1.
	sb.Add(createStats(1))
	expectSize(t, sb, 1)
	expectFirstN(t, sb, []int32{1})

	// Fill the buffer.
	for i := 1; i <= 5; i++ {
		expectSize(t, sb, i)
		sb.Add(createStats(int32(i)))
	}
	expectSize(t, sb, 5)
	expectFirstN(t, sb, []int32{1, 2, 3, 4, 5})

	// Add more than is available in the buffer
	sb.Add(createStats(6))
	expectSize(t, sb, 5)
	expectFirstN(t, sb, []int32{2, 3, 4, 5, 6})

	// Replace all elements.
	for i := 7; i <= 10; i++ {
		sb.Add(createStats(int32(i)))
	}
	expectSize(t, sb, 5)
	expectFirstN(t, sb, []int32{6, 7, 8, 9, 10})
}

func TestGet(t *testing.T) {
	sb := NewStatsBuffer(5)
	sb.Add(createStats(1))
	sb.Add(createStats(2))
	sb.Add(createStats(3))
	expectSize(t, sb, 3)
	expectFirstN(t, sb, []int32{1, 2, 3})

	expectElement(t, sb.Get(0), 3)
	expectElement(t, sb.Get(1), 2)
	expectElement(t, sb.Get(2), 1)
}

func TestInTimeRange(t *testing.T) {
	sb := NewStatsBuffer(5)
	assert := assert.New(t)

	var empty time.Time

	// No elements.
	assert.Empty(sb.InTimeRange(createTime(0), createTime(5), 10))
	assert.Empty(sb.InTimeRange(createTime(0), empty, 10))
	assert.Empty(sb.InTimeRange(empty, createTime(5), 10))
	assert.Empty(sb.InTimeRange(empty, empty, 10))

	// One element.
	sb.Add(createStats(1))
	expectSize(t, sb, 1)
	expectElements(t, sb.InTimeRange(createTime(0), createTime(5), 10), []int32{1})
	expectElements(t, sb.InTimeRange(createTime(1), createTime(5), 10), []int32{1})
	expectElements(t, sb.InTimeRange(createTime(0), createTime(1), 10), []int32{1})
	expectElements(t, sb.InTimeRange(createTime(1), createTime(1), 10), []int32{1})
	assert.Empty(sb.InTimeRange(createTime(2), createTime(5), 10))

	// Two element.
	sb.Add(createStats(2))
	expectSize(t, sb, 2)
	expectElements(t, sb.InTimeRange(createTime(0), createTime(5), 10), []int32{1, 2})
	expectElements(t, sb.InTimeRange(createTime(1), createTime(5), 10), []int32{1, 2})
	expectElements(t, sb.InTimeRange(createTime(0), createTime(2), 10), []int32{1, 2})
	expectElements(t, sb.InTimeRange(createTime(1), createTime(2), 10), []int32{1, 2})
	expectElements(t, sb.InTimeRange(createTime(1), createTime(1), 10), []int32{1})
	expectElements(t, sb.InTimeRange(createTime(2), createTime(2), 10), []int32{2})
	assert.Empty(sb.InTimeRange(createTime(3), createTime(5), 10))

	// Many elements.
	sb.Add(createStats(3))
	sb.Add(createStats(4))
	expectSize(t, sb, 4)
	expectElements(t, sb.InTimeRange(createTime(0), createTime(5), 10), []int32{1, 2, 3, 4})
	expectElements(t, sb.InTimeRange(createTime(0), createTime(5), 10), []int32{1, 2, 3, 4})
	expectElements(t, sb.InTimeRange(createTime(1), createTime(5), 10), []int32{1, 2, 3, 4})
	expectElements(t, sb.InTimeRange(createTime(0), createTime(4), 10), []int32{1, 2, 3, 4})
	expectElements(t, sb.InTimeRange(createTime(1), createTime(4), 10), []int32{1, 2, 3, 4})
	expectElements(t, sb.InTimeRange(createTime(0), createTime(2), 10), []int32{1, 2})
	expectElements(t, sb.InTimeRange(createTime(1), createTime(2), 10), []int32{1, 2})
	expectElements(t, sb.InTimeRange(createTime(2), createTime(3), 10), []int32{2, 3})
	expectElements(t, sb.InTimeRange(createTime(3), createTime(4), 10), []int32{3, 4})
	expectElements(t, sb.InTimeRange(createTime(3), createTime(5), 10), []int32{3, 4})
	assert.Empty(sb.InTimeRange(createTime(5), createTime(5), 10))

	// No start time.
	expectElements(t, sb.InTimeRange(empty, createTime(5), 10), []int32{1, 2, 3, 4})
	expectElements(t, sb.InTimeRange(empty, createTime(4), 10), []int32{1, 2, 3, 4})
	expectElements(t, sb.InTimeRange(empty, createTime(3), 10), []int32{1, 2, 3})
	expectElements(t, sb.InTimeRange(empty, createTime(2), 10), []int32{1, 2})
	expectElements(t, sb.InTimeRange(empty, createTime(1), 10), []int32{1})

	// No end time.
	expectElements(t, sb.InTimeRange(createTime(0), empty, 10), []int32{1, 2, 3, 4})
	expectElements(t, sb.InTimeRange(createTime(1), empty, 10), []int32{1, 2, 3, 4})
	expectElements(t, sb.InTimeRange(createTime(2), empty, 10), []int32{2, 3, 4})
	expectElements(t, sb.InTimeRange(createTime(3), empty, 10), []int32{3, 4})
	expectElements(t, sb.InTimeRange(createTime(4), empty, 10), []int32{4})

	// No start or end time.
	expectElements(t, sb.InTimeRange(empty, empty, 10), []int32{1, 2, 3, 4})

	// Start after data.
	assert.Empty(sb.InTimeRange(createTime(5), createTime(5), 10))
	assert.Empty(sb.InTimeRange(createTime(5), empty, 10))

	// End before data.
	assert.Empty(sb.InTimeRange(createTime(0), createTime(0), 10))
	assert.Empty(sb.InTimeRange(empty, createTime(0), 10))
}

func TestInTimeRangeWithLimit(t *testing.T) {
	sb := NewStatsBuffer(5)
	sb.Add(createStats(1))
	sb.Add(createStats(2))
	sb.Add(createStats(3))
	sb.Add(createStats(4))
	expectSize(t, sb, 4)

	var empty time.Time

	// Limit cuts off from latest timestamp.
	expectElements(t, sb.InTimeRange(empty, empty, 4), []int32{1, 2, 3, 4})
	expectElements(t, sb.InTimeRange(empty, empty, 3), []int32{2, 3, 4})
	expectElements(t, sb.InTimeRange(empty, empty, 2), []int32{3, 4})
	expectElements(t, sb.InTimeRange(empty, empty, 1), []int32{4})
	assert.Empty(t, sb.InTimeRange(empty, empty, 0))
}
