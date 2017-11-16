// +build windows

/*
Copyright 2017 The Kubernetes Authors.

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

package winstats

import (
	"errors"
	"fmt"
	"time"
	"unsafe"

	"github.com/lxn/win"
)

const (
	cpuQuery                  = "\\Processor(_Total)\\% Processor Time"
	memoryPrivWorkingSetQuery = "\\Process(_Total)\\Working Set - Private"
	memoryCommittedBytesQuery = "\\Memory\\Committed Bytes"
	// Perf counters are updated every second. This is the same as the default cadvisor collection period
	// see https://github.com/google/cadvisor/blob/master/docs/runtime_options.md#housekeeping
	perfCounterUpdatePeriod = 1 * time.Second
)

type perfCounter struct {
	queryHandle   win.PDH_HQUERY
	counterHandle win.PDH_HCOUNTER
}

func newPerfCounter(counter string) (*perfCounter, error) {
	var queryHandle win.PDH_HQUERY
	var counterHandle win.PDH_HCOUNTER

	ret := win.PdhOpenQuery(0, 0, &queryHandle)
	if ret != win.ERROR_SUCCESS {
		return nil, errors.New("unable to open query through DLL call")
	}

	ret = win.PdhValidatePath(counter)
	if ret != win.ERROR_SUCCESS {
		return nil, fmt.Errorf("unable to valid path to counter. Error code is %x", ret)
	}

	ret = win.PdhAddEnglishCounter(queryHandle, counter, 0, &counterHandle)
	if ret != win.ERROR_SUCCESS {
		return nil, fmt.Errorf("unable to add process counter. Error code is %x", ret)
	}

	ret = win.PdhCollectQueryData(queryHandle)
	if ret != win.ERROR_SUCCESS {
		return nil, fmt.Errorf("unable to collect data from counter. Error code is %x", ret)
	}

	return &perfCounter{
		queryHandle:   queryHandle,
		counterHandle: counterHandle,
	}, nil
}

func (p *perfCounter) getData() (uint64, error) {
	ret := win.PdhCollectQueryData(p.queryHandle)
	if ret != win.ERROR_SUCCESS {
		return 0, fmt.Errorf("unable to collect data from counter. Error code is %x", ret)
	}

	var bufSize, bufCount uint32
	var size = uint32(unsafe.Sizeof(win.PDH_FMT_COUNTERVALUE_ITEM_DOUBLE{}))
	var emptyBuf [1]win.PDH_FMT_COUNTERVALUE_ITEM_DOUBLE // need at least 1 addressable null ptr.
	var data uint64

	ret = win.PdhGetFormattedCounterArrayDouble(p.counterHandle, &bufSize, &bufCount, &emptyBuf[0])
	if ret != win.PDH_MORE_DATA {
		return 0, fmt.Errorf("unable to collect data from counter. Error code is %x", ret)
	}

	filledBuf := make([]win.PDH_FMT_COUNTERVALUE_ITEM_DOUBLE, bufCount*size)
	ret = win.PdhGetFormattedCounterArrayDouble(p.counterHandle, &bufSize, &bufCount, &filledBuf[0])
	if ret != win.ERROR_SUCCESS {
		return 0, fmt.Errorf("unable to collect data from counter. Error code is %x", ret)
	}

	for i := 0; i < int(bufCount); i++ {
		c := filledBuf[i]
		data = uint64(c.FmtValue.DoubleValue)
	}

	return data, nil
}
