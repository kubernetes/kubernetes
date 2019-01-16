/*
Copyright 2019 The Kubernetes Authors.

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

package socketmask

import (
	"bytes"
	"k8s.io/klog"
	"math"
	"strconv"
	"strings"
)

// SocketMask represents the NUMA affinity of a socket.
type SocketMask []int64

// NewSocketMask creates a new socket mask.
func NewSocketMask(Mask []int64) SocketMask {
	return Mask
}

// GetSocketMask calculates the socket mask.
func (sm SocketMask) GetSocketMask(socketMask []SocketMask, maskHolder []string, count int) (SocketMask, []string) {
	var socketAffinityInt64 [][]int64
	for r := range socketMask {
		socketAffinityVals := []int64(socketMask[r])
		socketAffinityInt64 = append(socketAffinityInt64, socketAffinityVals)
	}
	if count == 0 {
		maskHolder = buildMaskHolder(socketAffinityInt64)
	}
	klog.V(4).Infof("[socketmask] MaskHolder : %v", maskHolder)
	klog.V(4).Infof("[socketmask] %v is passed into arrange function", socketMask)
	arrangedMask := arrangeMask(socketAffinityInt64)
	newMask := getTopologyAffinity(arrangedMask, maskHolder)
	klog.V(4).Infof("[socketmask] New Mask after getTopologyAffinity (new mask) : %v ", newMask)
	finalMaskValue := parseMask(newMask)
	klog.V(4).Infof("[socketmask] Mask []Int64 (finalMaskValue): %v", finalMaskValue)
	maskHolder = newMask
	klog.V(4).Infof("[socketmask] New MaskHolder: %v", maskHolder)
	return SocketMask(finalMaskValue), maskHolder
}

func buildMaskHolder(mask [][]int64) []string {
	outerLen := len(mask)
	var innerLen int
	for i := 0; i < outerLen; i++ {
		if innerLen < len(mask[i]) {
			innerLen = len(mask[i])
		}
	}
	var maskHolder []string
	var buffer bytes.Buffer
	var i, j int = 0, 0
	for i = 0; i < outerLen; i++ {
		for j = 0; j < innerLen; j++ {
			buffer.WriteString("1")
		}
		maskHolder = append(maskHolder, buffer.String())
		buffer.Reset()
	}
	return maskHolder
}

func getTopologyAffinity(arrangedMask, maskHolder []string) []string {
	var topologyTemp []string
	for i := 0; i < len(maskHolder); i++ {
		for j := 0; j < len(arrangedMask); j++ {
			tempStr := andOperation(maskHolder[i], arrangedMask[j])
			if strings.Contains(tempStr, "1") {
				topologyTemp = append(topologyTemp, tempStr)
			}
		}
	}
	duplicates := map[string]bool{}
	for v := range topologyTemp {
		duplicates[topologyTemp[v]] = true
	}
	// Place all keys from the map into a slice.
	topologyResult := []string{}
	for key := range duplicates {
		topologyResult = append(topologyResult, key)
	}

	return topologyResult
}

func parseMask(mask []string) []int64 {
	var maskStr string
	min := strings.Count(mask[0], "1")
	var num, index int

	for i := 0; i < len(mask); i++ {
		num = strings.Count(mask[i], "1")
		if num < min {
			min = num
			index = i
		}
		maskStr = mask[index]
	}
	var maskInt []int64
	for _, char := range maskStr {
		convertedStr, err := strconv.Atoi(string(char))
		if err != nil {
			klog.V(4).Infof("could not convert mask character: %v", err)
			return maskInt
		}
		maskInt = append(maskInt, int64(convertedStr))
	}
	klog.V(4).Infof("[socketmask] Mask Int in Parse Mask: %v", maskInt)
	return maskInt
}

func arrangeMask(mask [][]int64) []string {
	var socketStr []string
	var bufferNew bytes.Buffer
	outerLen := len(mask)
	innerLen := len(mask[0])
	for i := 0; i < outerLen; i++ {
		for j := 0; j < innerLen; j++ {
			if mask[i][j] == 1 {
				bufferNew.WriteString("1")
			} else if mask[i][j] == 0 {
				bufferNew.WriteString("0")
			}
		}
		socketStr = append(socketStr, bufferNew.String())
		bufferNew.Reset()
	}
	return socketStr
}

func andOperation(val1, val2 string) string {
	l1, l2 := len(val1), len(val2)
	//compare lengths of strings - pad shortest with trailing zeros
	if l1 != l2 {
		// Get the bit difference
		var num int
		diff := math.Abs(float64(l1) - float64(l2))
		num = int(diff)
		if l1 < l2 {
			val1 = val1 + strings.Repeat("0", num)
		} else {
			val2 = val2 + strings.Repeat("0", num)
		}
	}
	length := len(val1)
	byteArr := make([]byte, length)
	for i := 0; i < length; i++ {
		byteArr[i] = val1[i] & val2[i]
	}

	return string(byteArr[:])
}
