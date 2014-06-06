/*
Copyright 2014 Google Inc. All rights reserved.

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
package util

import (
	"encoding/json"
	"log"
	"time"
)

// Simply catches a crash and logs an error. Meant to be called via defer.
func HandleCrash() {
	r := recover()
	if r != nil {
		log.Printf("Recovered from panic: %#v", r)
	}
}

// Loops forever running f every d.  Catches any panics, and keeps going.
func Forever(f func(), period time.Duration) {
	for {
		func() {
			defer HandleCrash()
			f()
		}()
		time.Sleep(period)
	}
}

// Returns o marshalled as a JSON string, ignoring any errors.
func MakeJSONString(o interface{}) string {
	data, _ := json.Marshal(o)
	return string(data)
}
