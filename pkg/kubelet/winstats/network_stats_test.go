//go:build windows
// +build windows

/*
Copyright 2023 The Kubernetes Authors.

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
	"testing"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
)

const fakeAdapterName = "fake-adapter"

type fakePerfCounterImpl struct {
	// Returned name.
	name string
	// Returned value.
	value uint64
	// If the perfCounter should raise an error.
	raiseError bool
}

func (p *fakePerfCounterImpl) getData() (uint64, error) {
	if p.raiseError {
		return 0, errors.New("Expected getData error.")
	}
	return p.value, nil
}

func (p *fakePerfCounterImpl) getDataList() (map[string]uint64, error) {
	if p.raiseError {
		return nil, errors.New("Expected getDataList error.")
	}

	data := make(map[string]uint64)
	data[p.name] = p.value
	return data, nil
}

func newFakedNetworkCounters(raiseError bool) *networkCounter {
	counters := make([]*fakePerfCounterImpl, 8)
	for i := 0; i < 8; i++ {
		counters[i] = &fakePerfCounterImpl{
			name:       fakeAdapterName,
			value:      1,
			raiseError: raiseError,
		}
	}
	return &networkCounter{
		packetsReceivedPerSecondCounter: counters[0],
		packetsSentPerSecondCounter:     counters[1],
		bytesReceivedPerSecondCounter:   counters[2],
		bytesSentPerSecondCounter:       counters[3],
		packetsReceivedDiscardedCounter: counters[4],
		packetsReceivedErrorsCounter:    counters[5],
		packetsOutboundDiscardedCounter: counters[6],
		packetsOutboundErrorsCounter:    counters[7],
		adapterStats:                    map[string]cadvisorapi.InterfaceStats{},
	}
}

func TestNewNetworkCounters(t *testing.T) {
	_, err := newNetworkCounters()
	assert.NoError(t, err)
}

func TestNetworkGetData(t *testing.T) {
	netCounter := newFakedNetworkCounters(false)

	// Add a net adapter that no longer exists in the adapterStats cache. It will
	// have to be cleaned up after processing the data.
	netCounter.adapterStats["other-fake-adapter"] = cadvisorapi.InterfaceStats{}

	data, err := netCounter.getData()
	assert.NoError(t, err)

	// Make sure that we only have data from a single net adapter.
	expectedStats := cadvisorapi.InterfaceStats{
		Name:      fakeAdapterName,
		RxPackets: 1,
		TxPackets: 1,
		RxBytes:   1,
		TxBytes:   1,
		RxDropped: 1,
		RxErrors:  1,
		TxDropped: 1,
		TxErrors:  1,
	}
	assert.Equal(t, []cadvisorapi.InterfaceStats{expectedStats}, data)

	// The returned data is cumulative, so the resulting values should be double on a second call.
	data, err = netCounter.getData()
	assert.NoError(t, err)
	expectedStats = cadvisorapi.InterfaceStats{
		Name:      fakeAdapterName,
		RxPackets: 2,
		TxPackets: 2,
		RxBytes:   2,
		TxBytes:   2,
		RxDropped: 1,
		RxErrors:  1,
		TxDropped: 1,
		TxErrors:  1,
	}
	assert.Equal(t, []cadvisorapi.InterfaceStats{expectedStats}, data)
}

func TestNetworkGetDataFailures(t *testing.T) {
	netCounter := newFakedNetworkCounters(true)

	_, err := netCounter.getData()
	expectedMsg := "Expected getDataList error."
	if err == nil || err.Error() != expectedMsg {
		t.Fatalf("expected error message `%s` but got `%v`", expectedMsg, err)
	}

	_, err = netCounter.getData()
	netCounter.packetsReceivedPerSecondCounter.(*fakePerfCounterImpl).raiseError = false
	if err == nil || err.Error() != expectedMsg {
		t.Fatalf("expected error message `%s` but got `%v`", expectedMsg, err)
	}

	_, err = netCounter.getData()
	netCounter.packetsSentPerSecondCounter.(*fakePerfCounterImpl).raiseError = false
	if err == nil || err.Error() != expectedMsg {
		t.Fatalf("expected error message `%s` but got `%v`", expectedMsg, err)
	}

	_, err = netCounter.getData()
	netCounter.bytesReceivedPerSecondCounter.(*fakePerfCounterImpl).raiseError = false
	if err == nil || err.Error() != expectedMsg {
		t.Fatalf("expected error message `%s` but got `%v`", expectedMsg, err)
	}

	_, err = netCounter.getData()
	netCounter.bytesSentPerSecondCounter.(*fakePerfCounterImpl).raiseError = false
	if err == nil || err.Error() != expectedMsg {
		t.Fatalf("expected error message `%s` but got `%v`", expectedMsg, err)
	}

	_, err = netCounter.getData()
	netCounter.packetsReceivedDiscardedCounter.(*fakePerfCounterImpl).raiseError = false
	if err == nil || err.Error() != expectedMsg {
		t.Fatalf("expected error message `%s` but got `%v`", expectedMsg, err)
	}

	_, err = netCounter.getData()
	netCounter.packetsReceivedErrorsCounter.(*fakePerfCounterImpl).raiseError = false
	if err == nil || err.Error() != expectedMsg {
		t.Fatalf("expected error message `%s` but got `%v`", expectedMsg, err)
	}

	_, err = netCounter.getData()
	netCounter.packetsOutboundDiscardedCounter.(*fakePerfCounterImpl).raiseError = false
	if err == nil || err.Error() != expectedMsg {
		t.Fatalf("expected error message `%s` but got `%v`", expectedMsg, err)
	}
}
