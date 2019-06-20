// Copyright 2015 Light Code Labs, LLC
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

package telemetry

import (
	"fmt"
	"hash/fnv"
	"log"
	"strings"

	"github.com/google/uuid"
)

// Init initializes this package so that it may
// be used. Do not call this function more than
// once. Init panics if it is called more than
// once or if the UUID value is empty. Once this
// function is called, the rest of the package
// may safely be used. If this function is not
// called, the collector functions may still be
// invoked, but they will be no-ops.
//
// Any metrics keys that are passed in the second
// argument will be permanently disabled for the
// lifetime of the process.
func Init(instanceID uuid.UUID, disabledMetricsKeys []string) {
	if enabled {
		panic("already initialized")
	}
	if str := instanceID.String(); str == "" ||
		str == "00000000-0000-0000-0000-000000000000" {
		panic("empty UUID")
	}
	instanceUUID = instanceID
	disabledMetricsMu.Lock()
	for _, key := range disabledMetricsKeys {
		disabledMetrics[strings.TrimSpace(key)] = false
	}
	disabledMetricsMu.Unlock()
	enabled = true
}

// StartEmitting sends the current payload and begins the
// transmission cycle for updates. This is the first
// update sent, and future ones will be sent until
// StopEmitting is called.
//
// This function is non-blocking (it spawns a new goroutine).
//
// This function panics if it was called more than once.
// It is a no-op if this package was not initialized.
func StartEmitting() {
	if !enabled {
		return
	}
	updateTimerMu.Lock()
	if updateTimer != nil {
		updateTimerMu.Unlock()
		panic("updates already started")
	}
	updateTimerMu.Unlock()
	updateMu.Lock()
	if updating {
		updateMu.Unlock()
		panic("update already in progress")
	}
	updateMu.Unlock()
	go logEmit(false)
}

// StopEmitting sends the current payload and terminates
// the update cycle. No more updates will be sent.
//
// It is a no-op if the package was never initialized
// or if emitting was never started.
//
// NOTE: This function is blocking. Run in a goroutine if
// you want to guarantee no blocking at critical times
// like exiting the program.
func StopEmitting() {
	if !enabled {
		return
	}
	updateTimerMu.Lock()
	if updateTimer == nil {
		updateTimerMu.Unlock()
		return
	}
	updateTimerMu.Unlock()
	logEmit(true) // likely too early; may take minutes to return
}

// Reset empties the current payload buffer.
func Reset() {
	resetBuffer()
}

// Set puts a value in the buffer to be included
// in the next emission. It overwrites any
// previous value.
//
// This function is safe for multiple goroutines,
// and it is recommended to call this using the
// go keyword after the call to SendHello so it
// doesn't block crucial code.
func Set(key string, val interface{}) {
	if !enabled || isDisabled(key) {
		return
	}
	bufferMu.Lock()
	if _, ok := buffer[key]; !ok {
		if bufferItemCount >= maxBufferItems {
			bufferMu.Unlock()
			return
		}
		bufferItemCount++
	}
	buffer[key] = val
	bufferMu.Unlock()
}

// SetNested puts a value in the buffer to be included
// in the next emission, nested under the top-level key
// as subkey. It overwrites any previous value.
//
// This function is safe for multiple goroutines,
// and it is recommended to call this using the
// go keyword after the call to SendHello so it
// doesn't block crucial code.
func SetNested(key, subkey string, val interface{}) {
	if !enabled || isDisabled(key) {
		return
	}
	bufferMu.Lock()
	if topLevel, ok1 := buffer[key]; ok1 {
		topLevelMap, ok2 := topLevel.(map[string]interface{})
		if !ok2 {
			bufferMu.Unlock()
			log.Printf("[PANIC] Telemetry: key %s is already used for non-nested-map value", key)
			return
		}
		if _, ok3 := topLevelMap[subkey]; !ok3 {
			// don't exceed max buffer size
			if bufferItemCount >= maxBufferItems {
				bufferMu.Unlock()
				return
			}
			bufferItemCount++
		}
		topLevelMap[subkey] = val
	} else {
		// don't exceed max buffer size
		if bufferItemCount >= maxBufferItems {
			bufferMu.Unlock()
			return
		}
		bufferItemCount++
		buffer[key] = map[string]interface{}{subkey: val}
	}
	bufferMu.Unlock()
}

// Append appends value to a list named key.
// If key is new, a new list will be created.
// If key maps to a type that is not a list,
// a panic is logged, and this is a no-op.
func Append(key string, value interface{}) {
	if !enabled || isDisabled(key) {
		return
	}
	bufferMu.Lock()
	if bufferItemCount >= maxBufferItems {
		bufferMu.Unlock()
		return
	}
	// TODO: Test this...
	bufVal, inBuffer := buffer[key]
	sliceVal, sliceOk := bufVal.([]interface{})
	if inBuffer && !sliceOk {
		bufferMu.Unlock()
		log.Printf("[PANIC] Telemetry: key %s already used for non-slice value", key)
		return
	}
	if sliceVal == nil {
		buffer[key] = []interface{}{value}
	} else if sliceOk {
		buffer[key] = append(sliceVal, value)
	}
	bufferItemCount++
	bufferMu.Unlock()
}

// AppendUnique adds value to a set named key.
// Set items are unordered. Values in the set
// are unique, but how many times they are
// appended is counted. The value must be
// hashable.
//
// If key is new, a new set will be created for
// values with that key. If key maps to a type
// that is not a counting set, a panic is logged,
// and this is a no-op.
func AppendUnique(key string, value interface{}) {
	if !enabled || isDisabled(key) {
		return
	}
	bufferMu.Lock()
	bufVal, inBuffer := buffer[key]
	setVal, setOk := bufVal.(countingSet)
	if inBuffer && !setOk {
		bufferMu.Unlock()
		log.Printf("[PANIC] Telemetry: key %s already used for non-counting-set value", key)
		return
	}
	if setVal == nil {
		// ensure the buffer is not too full, then add new unique value
		if bufferItemCount >= maxBufferItems {
			bufferMu.Unlock()
			return
		}
		buffer[key] = countingSet{value: 1}
		bufferItemCount++
	} else if setOk {
		// unique value already exists, so just increment counter
		setVal[value]++
	}
	bufferMu.Unlock()
}

// Add adds amount to a value named key.
// If it does not exist, it is created with
// a value of 1. If key maps to a type that
// is not an integer, a panic is logged,
// and this is a no-op.
func Add(key string, amount int) {
	atomicAdd(key, amount)
}

// Increment is a shortcut for Add(key, 1)
func Increment(key string) {
	atomicAdd(key, 1)
}

// atomicAdd adds amount (negative to subtract)
// to key.
func atomicAdd(key string, amount int) {
	if !enabled || isDisabled(key) {
		return
	}
	bufferMu.Lock()
	bufVal, inBuffer := buffer[key]
	intVal, intOk := bufVal.(int)
	if inBuffer && !intOk {
		bufferMu.Unlock()
		log.Printf("[PANIC] Telemetry: key %s already used for non-integer value", key)
		return
	}
	if !inBuffer {
		if bufferItemCount >= maxBufferItems {
			bufferMu.Unlock()
			return
		}
		bufferItemCount++
	}
	buffer[key] = intVal + amount
	bufferMu.Unlock()
}

// FastHash hashes input using a 32-bit hashing algorithm
// that is fast, and returns the hash as a hex-encoded string.
// Do not use this for cryptographic purposes.
func FastHash(input []byte) string {
	h := fnv.New32a()
	if _, err := h.Write(input); err != nil {
		log.Println("[ERROR] failed to write bytes: ", err)
	}

	return fmt.Sprintf("%x", h.Sum32())
}

// isDisabled returns whether key is
// a disabled metric key. ALL collection
// functions should call this and not
// save the value if this returns true.
func isDisabled(key string) bool {
	// for keys that are augmented with data, such as
	// "tls_client_hello_ua:<hash>", just
	// check the prefix "tls_client_hello_ua"
	checkKey := key
	if idx := strings.Index(key, ":"); idx > -1 {
		checkKey = key[:idx]
	}

	disabledMetricsMu.RLock()
	_, ok := disabledMetrics[checkKey]
	disabledMetricsMu.RUnlock()
	return ok
}
