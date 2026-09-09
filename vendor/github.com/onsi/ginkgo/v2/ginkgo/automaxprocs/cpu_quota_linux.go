// Copyright (c) 2017 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//go:build linux
// +build linux

package automaxprocs

import (
	"errors"
)

// CPUQuotaToGOMAXPROCS converts the CPU quota applied to the calling process
// to a valid GOMAXPROCS value. The quota is converted from float to int using round.
// If round == nil, DefaultRoundFunc is used.
func CPUQuotaToGOMAXPROCS(minValue int, round func(v float64) int) (int, CPUQuotaStatus, error) {
	if round == nil {
		round = DefaultRoundFunc
	}
	cgroups, err := _newQueryer()
	if err != nil {
		return -1, CPUQuotaUndefined, err
	}

	quota, defined, err := cgroups.CPUQuota()
	if !defined || err != nil {
		return -1, CPUQuotaUndefined, err
	}

	maxProcs := round(quota)
	if minValue > 0 && maxProcs < minValue {
		return minValue, CPUQuotaMinUsed, nil
	}
	return maxProcs, CPUQuotaUsed, nil
}

type queryer interface {
	CPUQuota() (float64, bool, error)
}

var (
	_newCgroups2 = NewCGroups2ForCurrentProcess
	_newCgroups  = NewCGroupsForCurrentProcess
	_newQueryer  = newQueryer
)

func newQueryer() (queryer, error) {
	cgroups, err := _newCgroups2()
	if err == nil {
		return cgroups, nil
	}
	if errors.Is(err, ErrNotV2) {
		return _newCgroups()
	}
	return nil, err
}
