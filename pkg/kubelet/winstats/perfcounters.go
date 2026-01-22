//go:build windows

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

	"github.com/JeffAshton/win_pdh"
)

const (
	cpuQuery                  = "\\Processor(_Total)\\% Processor Time"
	memoryPrivWorkingSetQuery = "\\Process(_Total)\\Working Set - Private"
	memoryCommittedBytesQuery = "\\Memory\\Committed Bytes"
	// Perf counters are updated 10 seconds. This is the same as the default cadvisor housekeeping interval
	// set at https://github.com/kubernetes/kubernetes/blob/master/pkg/kubelet/cadvisor/cadvisor_linux.go
	perfCounterUpdatePeriod = 10 * time.Second
	// defaultCachePeriod is the default cache period for each cpuUsage.
	// This matches with the cadvisor setting and the time interval we use for containers.
	// see https://github.com/kubernetes/kubernetes/blob/master/pkg/kubelet/cadvisor/cadvisor_linux.go#L63
	defaultCachePeriod = 10 * time.Second
)

type perfCounter interface {
	getData() (uint64, error)
	getDataList() (map[string]uint64, error)
}

type perfCounterImpl struct {
	queryHandle   win_pdh.PDH_HQUERY
	counterHandle win_pdh.PDH_HCOUNTER
}

func newPerfCounter(counter string) (perfCounter, error) {
	var queryHandle win_pdh.PDH_HQUERY
	var counterHandle win_pdh.PDH_HCOUNTER

	ret := win_pdh.PdhOpenQuery(0, 0, &queryHandle)
	if ret != win_pdh.ERROR_SUCCESS {
		return nil, errors.New("unable to open query through DLL call")
	}

	ret = win_pdh.PdhAddEnglishCounter(queryHandle, counter, 0, &counterHandle)
	if ret != win_pdh.ERROR_SUCCESS {
		return nil, fmt.Errorf("unable to add process counter: %s. Error code is %x", counter, ret)
	}

	ret = win_pdh.PdhCollectQueryData(queryHandle)
	if ret != win_pdh.ERROR_SUCCESS {
		return nil, fmt.Errorf("unable to collect data from counter. Error code is %x", ret)
	}

	return &perfCounterImpl{
		queryHandle:   queryHandle,
		counterHandle: counterHandle,
	}, nil
}

// getData is used for getting data without * in counter name.
func (p *perfCounterImpl) getData() (uint64, error) {
	filledBuf, bufCount, err := p.getQueriedData()
	if err != nil {
		return 0, err
	}

	var data uint64 = 0
	for i := 0; i < int(bufCount); i++ {
		c := filledBuf[i]
		data = uint64(c.FmtValue.DoubleValue)
	}

	return data, nil
}

// getDataList is used for getting data with * in counter name.
func (p *perfCounterImpl) getDataList() (map[string]uint64, error) {
	filledBuf, bufCount, err := p.getQueriedData()
	if err != nil {
		return nil, err
	}

	data := map[string]uint64{}
	for i := 0; i < int(bufCount); i++ {
		c := filledBuf[i]
		value := uint64(c.FmtValue.DoubleValue)
		name := win_pdh.UTF16PtrToString(c.SzName)
		data[name] = value
	}

	return data, nil
}

// getQueriedData is used for getting data using the given query handle.
func (p *perfCounterImpl) getQueriedData() ([]win_pdh.PDH_FMT_COUNTERVALUE_ITEM_DOUBLE, uint32, error) {
	ret := win_pdh.PdhCollectQueryData(p.queryHandle)
	if ret != win_pdh.ERROR_SUCCESS {
		return nil, 0, fmt.Errorf("unable to collect data from counter. Error code is %x", ret)
	}

	var bufSize, bufCount uint32
	var size = uint32(unsafe.Sizeof(win_pdh.PDH_FMT_COUNTERVALUE_ITEM_DOUBLE{}))
	var emptyBuf [1]win_pdh.PDH_FMT_COUNTERVALUE_ITEM_DOUBLE // need at least 1 addressable null ptr.

	ret = win_pdh.PdhGetFormattedCounterArrayDouble(p.counterHandle, &bufSize, &bufCount, &emptyBuf[0])
	if ret != win_pdh.PDH_MORE_DATA {
		return nil, 0, fmt.Errorf("unable to collect data from counter. Error code is %x", ret)
	}

	filledBuf := make([]win_pdh.PDH_FMT_COUNTERVALUE_ITEM_DOUBLE, bufCount*size)
	ret = win_pdh.PdhGetFormattedCounterArrayDouble(p.counterHandle, &bufSize, &bufCount, &filledBuf[0])
	if ret != win_pdh.ERROR_SUCCESS {
		return nil, 0, fmt.Errorf("unable to collect data from counter. Error code is %x", ret)
	}

	return filledBuf, bufCount, nil
}
