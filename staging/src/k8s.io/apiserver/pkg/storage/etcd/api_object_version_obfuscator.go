/*
Copyright 2018 The Kubernetes Authors.

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

package etcd

import (
	"encoding/binary"
	"hash/fnv"
)

// IdentityObfuscator implements Obfuscator.
type IdentityObfuscator struct{}

// NewIdentityObfuscator instantiates a IdentityObfuscator object.
func NewIdentityObfuscator() IdentityObfuscator {
	return IdentityObfuscator{}
}

// Decode implements Obfuscator.
func (i IdentityObfuscator) Decode(key string, clientResourceVersion uint64) uint64 {
	return clientResourceVersion
}

// Encode implements Obfuscator.
func (i IdentityObfuscator) Encode(key string, etcdResourceVersion uint64) uint64 {
	return etcdResourceVersion
}

// FeistelObfuscator implements Obfuscator.
type FeistelObfuscator struct {
	KeyScheduleFunc func(key string) []uint32
	RoundFunc       func(block uint32, roundKey uint32) uint32
}

// NewFeistelObfuscator instantiates a FeistelObfuscator object.
func NewFeistelObfuscator() FeistelObfuscator {
	return FeistelObfuscator{
		KeyScheduleFunc: HashKeySchedule,
		RoundFunc:       ProductRoundFunc,
	}
}

// HashKeySchedule takes the 128 hash of a string and breaks it
// into 4 32-bit keys to be used as a key schedule.
func HashKeySchedule(key string) []uint32 {
	hash := fnv.New128a()
	hash.Write([]byte(key))
	hashBytes := hash.Sum(make([]byte, 0))
	roundKeys := make([]uint32, 0)
	for i := 0; i < len(hashBytes); i += 4 {
		roundKey := binary.BigEndian.Uint32(hashBytes[i : i+4])
		roundKeys = append(roundKeys, roundKey)
	}
	return roundKeys
}

// ProductRoundFunc is a round function that just multiplies the values
// of the block and the roundKey.
func ProductRoundFunc(block uint32, roundKey uint32) uint32 {
	return block * roundKey
}

// Decode implements Obfuscator.
func (f FeistelObfuscator) Decode(key string, clientResourceVersion uint64) uint64 {
	left, right := splitUint(clientResourceVersion)
	keySchedule := f.KeyScheduleFunc(key)
	for i := len(keySchedule) - 1; i >= 0; i-- {
		roundKey := keySchedule[i]
		left, right = right^f.RoundFunc(left, roundKey), left
	}
	etcdResourceVersion := joinUint(left, right)
	return etcdResourceVersion
}

// Encode implements Obfuscator. For a FeistelObfuscator, it should always map from 0 to 0,
// and for two randomly selected uint64 values x and y, (x > y) should have no predictable
// relationship with (Encode(x) > Encode(y)).
func (f FeistelObfuscator) Encode(key string, etcdResourceVersion uint64) uint64 {
	left, right := splitUint(etcdResourceVersion)
	keySchedule := f.KeyScheduleFunc(key)
	for i := 0; i < len(keySchedule); i++ {
		roundKey := keySchedule[i]
		left, right = right, left^f.RoundFunc(right, roundKey)
	}
	clientResourceVersion := joinUint(left, right)
	return clientResourceVersion
}

func splitUint(in uint64) (uint32, uint32) {
	left := uint32(in >> 32)
	right := uint32(in & 0xffffffff)
	return left, right
}

func joinUint(left, right uint32) uint64 {
	return uint64(left)<<32 | uint64(right)
}
