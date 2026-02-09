//go:build !linux

/*
Copyright 2015 The Kubernetes Authors.

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

package oom

import (
	"errors"
)

var unsupportedErr = errors.New("setting OOM scores is unsupported in this build")

func NewOOMAdjuster() *OOMAdjuster {
	return &OOMAdjuster{
		ApplyOOMScoreAdj:          unsupportedApplyOOMScoreAdj,
		ApplyOOMScoreAdjContainer: unsupportedApplyOOMScoreAdjContainer,
	}
}

func unsupportedApplyOOMScoreAdj(pid int, oomScoreAdj int) error {
	return unsupportedErr
}

func unsupportedApplyOOMScoreAdjContainer(cgroupName string, oomScoreAdj, maxTries int) error {
	return unsupportedErr
}
