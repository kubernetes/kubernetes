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

package api

import (
	"io"
	"net/http"
	"reflect"
	"testing"
	"time"

	"github.com/google/cadvisor/events"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/assert"
)

// returns an http.Request pointer for an input url test string
func makeHTTPRequest(requestURL string, t *testing.T) *http.Request {
	dummyReader, _ := io.Pipe()
	r, err := http.NewRequest("GET", requestURL, dummyReader)
	assert.Nil(t, err)
	return r
}

func TestGetEventRequestBasicRequest(t *testing.T) {
	r := makeHTTPRequest("http://localhost:8080/api/v1.3/events?oom_events=true&stream=false&max_events=20", t)
	expectedQuery := events.NewRequest()
	expectedQuery.EventType = map[info.EventType]bool{
		info.EventOom: true,
	}
	expectedQuery.MaxEventsReturned = 20

	receivedQuery, stream, err := getEventRequest(r)

	if !reflect.DeepEqual(expectedQuery, receivedQuery) {
		t.Errorf("expected %#v but received %#v", expectedQuery, receivedQuery)
	}
	assert.False(t, stream)
	assert.Nil(t, err)
}

func TestGetEventEmptyRequest(t *testing.T) {
	r := makeHTTPRequest("", t)
	expectedQuery := events.NewRequest()

	receivedQuery, stream, err := getEventRequest(r)

	if !reflect.DeepEqual(expectedQuery, receivedQuery) {
		t.Errorf("expected %#v but received %#v", expectedQuery, receivedQuery)
	}
	assert.False(t, stream)
	assert.Nil(t, err)
}

func TestGetEventRequestDoubleArgument(t *testing.T) {
	r := makeHTTPRequest("http://localhost:8080/api/v1.3/events?stream=true&oom_events=true&oom_events=false", t)
	expectedQuery := events.NewRequest()
	expectedQuery.EventType = map[info.EventType]bool{
		info.EventOom: true,
	}

	receivedQuery, stream, err := getEventRequest(r)

	if !reflect.DeepEqual(expectedQuery, receivedQuery) {
		t.Errorf("expected %#v but received %#v", expectedQuery, receivedQuery)
	}
	assert.True(t, stream)
	assert.Nil(t, err)
}

func TestInstCpuStats(t *testing.T) {
	tests := []struct {
		last *info.ContainerStats
		cur  *info.ContainerStats
		want *v2.CpuInstStats
	}{
		// Last is missing
		{
			nil,
			&info.ContainerStats{},
			nil,
		},
		// Goes back in time
		{
			&info.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(time.Second),
			},
			&info.ContainerStats{
				Timestamp: time.Unix(100, 0),
			},
			nil,
		},
		// Zero time delta
		{
			&info.ContainerStats{
				Timestamp: time.Unix(100, 0),
			},
			&info.ContainerStats{
				Timestamp: time.Unix(100, 0),
			},
			nil,
		},
		// Unexpectedly small time delta
		{
			&info.ContainerStats{
				Timestamp: time.Unix(100, 0),
			},
			&info.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(30 * time.Millisecond),
			},
			nil,
		},
		// Different number of cpus
		{
			&info.ContainerStats{
				Timestamp: time.Unix(100, 0),
				Cpu: info.CpuStats{
					Usage: info.CpuUsage{
						PerCpu: []uint64{100, 200},
					},
				},
			},
			&info.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(time.Second),
				Cpu: info.CpuStats{
					Usage: info.CpuUsage{
						PerCpu: []uint64{100, 200, 300},
					},
				},
			},
			nil,
		},
		// Stat numbers decrease
		{
			&info.ContainerStats{
				Timestamp: time.Unix(100, 0),
				Cpu: info.CpuStats{
					Usage: info.CpuUsage{
						Total:  300,
						PerCpu: []uint64{100, 200},
						User:   250,
						System: 50,
					},
				},
			},
			&info.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(time.Second),
				Cpu: info.CpuStats{
					Usage: info.CpuUsage{
						Total:  200,
						PerCpu: []uint64{100, 100},
						User:   150,
						System: 50,
					},
				},
			},
			nil,
		},
		// One second elapsed
		{
			&info.ContainerStats{
				Timestamp: time.Unix(100, 0),
				Cpu: info.CpuStats{
					Usage: info.CpuUsage{
						Total:  300,
						PerCpu: []uint64{100, 200},
						User:   250,
						System: 50,
					},
				},
			},
			&info.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(time.Second),
				Cpu: info.CpuStats{
					Usage: info.CpuUsage{
						Total:  500,
						PerCpu: []uint64{200, 300},
						User:   400,
						System: 100,
					},
				},
			},
			&v2.CpuInstStats{
				Usage: v2.CpuInstUsage{
					Total:  200,
					PerCpu: []uint64{100, 100},
					User:   150,
					System: 50,
				},
			},
		},
		// Two seconds elapsed
		{
			&info.ContainerStats{
				Timestamp: time.Unix(100, 0),
				Cpu: info.CpuStats{
					Usage: info.CpuUsage{
						Total:  300,
						PerCpu: []uint64{100, 200},
						User:   250,
						System: 50,
					},
				},
			},
			&info.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(2 * time.Second),
				Cpu: info.CpuStats{
					Usage: info.CpuUsage{
						Total:  500,
						PerCpu: []uint64{200, 300},
						User:   400,
						System: 100,
					},
				},
			},
			&v2.CpuInstStats{
				Usage: v2.CpuInstUsage{
					Total:  100,
					PerCpu: []uint64{50, 50},
					User:   75,
					System: 25,
				},
			},
		},
	}
	for _, c := range tests {
		got, err := instCpuStats(c.last, c.cur)
		if err != nil {
			if c.want == nil {
				continue
			}
			t.Errorf("Unexpected error: %v", err)
		}
		assert.Equal(t, c.want, got)
	}
}
