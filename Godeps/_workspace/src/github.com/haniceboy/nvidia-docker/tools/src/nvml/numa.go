// Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.

package nvml

import (
	"bytes"
	"encoding/hex"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
)

type cpuSet [4]uint64

func (s cpuSet) equal(set cpuSet) bool {
	for i := range s {
		if s[i] != set[i] {
			return false
		}
	}
	return true
}

var cpuNodes []cpuSet

func getCPUNode(cpus cpuSet) (uint, error) {
	var err error

	if cpuNodes == nil {
		cpuNodes, err = parseCPUNodes()
		if err != nil {
			return 0, err
		}
	}
	for i := range cpuNodes {
		if cpuNodes[i].equal(cpus) {
			return uint(i), nil
		}
	}
	return 0, ErrCPUAffinity
}

func parseCPUNodes() (nodes []cpuSet, err error) {
	fn := func(p string, fi os.FileInfo, err error) error {
		var node cpuSet

		if err != nil {
			return err
		}
		if path.Base(p) != "cpumap" {
			return nil
		}
		hexa, err := ioutil.ReadFile(p)
		if err != nil {
			return err
		}

		// Hex chunks might be separated by commas, strip them off and decode the string
		hexa = bytes.Replace(hexa[:len(hexa)-1], []byte(","), nil, -1)
		if len(hexa)%2 != 0 {
			hexa = append([]byte("0"), hexa...)
		}
		b := make([]byte, len(hexa)/2)
		if _, err := hex.Decode(b, hexa); err != nil {
			return ErrCPUAffinity
		}

		// Fill up the CPU set starting from the end of the hex array
		for i := range b {
			if i/8 > len(node)-1 {
				break
			}
			node[i/8] |= uint64(b[len(b)-1-i]) << uint(i%8*8)
		}

		nodes = append(nodes, node)
		return nil
	}

	err = filepath.Walk("/sys/devices/system/node", fn)
	return
}
