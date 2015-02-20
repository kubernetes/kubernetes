// Copyright 2012 The Gorilla Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package context

import (
	"net/http"
	"sync"
	"time"
)

var (
	mutex sync.Mutex
	data  = make(map[*http.Request]map[interface{}]interface{})
	datat = make(map[*http.Request]int64)
)

// Set stores a value for a given key in a given request.
func Set(r *http.Request, key, val interface{}) {
	mutex.Lock()
	defer mutex.Unlock()
	if data[r] == nil {
		data[r] = make(map[interface{}]interface{})
		datat[r] = time.Now().Unix()
	}
	data[r][key] = val
}

// Get returns a value stored for a given key in a given request.
func Get(r *http.Request, key interface{}) interface{} {
	mutex.Lock()
	defer mutex.Unlock()
	if data[r] != nil {
		return data[r][key]
	}
	return nil
}

// Delete removes a value stored for a given key in a given request.
func Delete(r *http.Request, key interface{}) {
	mutex.Lock()
	defer mutex.Unlock()
	if data[r] != nil {
		delete(data[r], key)
	}
}

// Clear removes all values stored for a given request.
//
// This is usually called by a handler wrapper to clean up request
// variables at the end of a request lifetime. See ClearHandler().
func Clear(r *http.Request) {
	mutex.Lock()
	defer mutex.Unlock()
	clear(r)
}

// clear is Clear without the lock.
func clear(r *http.Request) {
	delete(data, r)
	delete(datat, r)
}

// Purge removes request data stored for longer than maxAge, in seconds.
// It returns the amount of requests removed.
//
// If maxAge <= 0, all request data is removed.
//
// This is only used for sanity check: in case context cleaning was not
// properly set some request data can be kept forever, consuming an increasing
// amount of memory. In case this is detected, Purge() must be called
// periodically until the problem is fixed.
func Purge(maxAge int) int {
	mutex.Lock()
	defer mutex.Unlock()
	count := 0
	if maxAge <= 0 {
		count = len(data)
		data = make(map[*http.Request]map[interface{}]interface{})
		datat = make(map[*http.Request]int64)
	} else {
		min := time.Now().Unix() - int64(maxAge)
		for r, _ := range data {
			if datat[r] < min {
				clear(r)
				count++
			}
		}
	}
	return count
}

// ClearHandler wraps an http.Handler and clears request values at the end
// of a request lifetime.
func ClearHandler(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer Clear(r)
		h.ServeHTTP(w, r)
	})
}
