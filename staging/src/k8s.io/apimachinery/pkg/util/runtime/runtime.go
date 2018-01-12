/*
Copyright 2014 The Kubernetes Authors.

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

package runtime

import (
	"fmt"
	"reflect"
	"regexp"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/agext/levenshtein"
	"github.com/golang/glog"
	"github.com/golang/groupcache/lru"

	"k8s.io/apimachinery/pkg/util/clock"
)

var (
	// ReallyCrash controls the behavior of HandleCrash and now defaults
	// true. It's still exposed so components can optionally set to false
	// to restore prior behavior.
	ReallyCrash = true
)

// PanicHandlers is a list of functions which will be invoked when a panic happens.
var PanicHandlers = []func(interface{}){logPanic}

// HandleCrash simply catches a crash and logs an error. Meant to be called via
// defer.  Additional context-specific handlers can be provided, and will be
// called in case of panic.  HandleCrash actually crashes, after calling the
// handlers and logging the panic message.
//
// TODO: remove this function. We are switching to a world where it's safe for
// apiserver to panic, since it will be restarted by kubelet. At the beginning
// of the Kubernetes project, nothing was going to restart apiserver and so
// catching panics was important. But it's actually much simpler for monitoring
// software if we just exit when an unexpected panic happens.
func HandleCrash(additionalHandlers ...func(interface{})) {
	if r := recover(); r != nil {
		for _, fn := range PanicHandlers {
			fn(r)
		}
		for _, fn := range additionalHandlers {
			fn(r)
		}
		if ReallyCrash {
			// Actually proceed to panic.
			panic(r)
		}
	}
}

// logPanic logs the caller tree when a panic occurs.
func logPanic(r interface{}) {
	glog.Errorf("Observed a panic: %#v (%v)\n%v", r, r, getCallers())
}

func getCallers() string {
	callers := ""
	for i := 0; true; i++ {
		_, file, line, ok := runtime.Caller(i)
		if !ok {
			break
		}
		callers = callers + fmt.Sprintf("%v:%v\n", file, line)
	}

	return callers
}

// these constants determine the default behavior of dedupingErrorHandler
const (
	// cacheSize determines how many "unique" errors to track (see errKey and dedupingErrorHandler.getVal below)
	cacheSize = 1000
	// errorDepth determines how many callers need to be skipped to find the stack of HandleError's caller
	// HandleError -> ErrorHandlers iteration -> dedupingErrorHandler.handleErr -> dedupingErrorHandler.getStackHandler -> dedupingErrorHandler.getStack
	errorDepth = 5
	// delta determines how long we can go in between logging the same error that we have seen multiple times
	// it mainly exists to handle the extreme ends of the error spectrum:
	// 1. a component that sends the same error very infrequently (we want to log these so we do not miss them)
	// 2. a component that sends the same error as fast as it can (we want to suppress most of these, but still see it every so often)
	delta = 5 * time.Minute
	// similar determines if two strings are close enough to be considered equal via levenshtein ratio
	// TODO: set this to something like 0.75 to actually have fuzzy matching
	// once we are comfortable with using this throughout the system
	similar = 1
)

// DedupingErrorHandler is exported to allow external callers to override dedupingErrorHandler.Similar
// TODO remove this global var once we have a useful default value for similar
var DedupingErrorHandler = newDedupingErrorHandler(cacheSize, errorDepth, delta, similar)

// ErrorHandlers is a list of functions which will be invoked when an unreturnable
// error occurs.
// TODO(lavalamp): for testability, this and the below HandleError function
// should be packaged up into a testable and reusable object.
var ErrorHandlers = []func(error){
	DedupingErrorHandler.handleErr,
	(&rudimentaryErrorBackoff{
		lastErrorTime: time.Now(),
		// 1ms was the number folks were able to stomach as a global rate limit.
		// If you need to log errors more than 1000 times a second you
		// should probably consider fixing your code instead. :)
		minPeriod: time.Millisecond,
	}).OnError,
}

// HandleError is a method to invoke when a non-user facing piece of code cannot
// return an error and needs to indicate it has been ignored. Invoking this method
// is preferable to logging the error - the default behavior is to log but the
// errors may be sent to a remote server for analysis.
func HandleError(err error) {
	// this is sometimes called with a nil error.  We probably shouldn't fail and should do nothing instead
	if err == nil {
		return
	}

	for _, fn := range ErrorHandlers {
		fn(err)
	}
}

// logError prints an error with the call stack of the location it was reported
func logError(err error) {
	glog.ErrorDepth(2, err)
}

type rudimentaryErrorBackoff struct {
	minPeriod time.Duration // immutable
	// TODO(lavalamp): use the clock for testability. Need to move that
	// package for that to be accessible here.
	lastErrorTimeLock sync.Mutex
	lastErrorTime     time.Time
}

// OnError will block if it is called more often than the embedded period time.
// This will prevent overly tight hot error loops.
func (r *rudimentaryErrorBackoff) OnError(error) {
	r.lastErrorTimeLock.Lock()
	defer r.lastErrorTimeLock.Unlock()
	d := time.Since(r.lastErrorTime)
	if d < r.minPeriod && d >= 0 {
		// If the time moves backwards for any reason, do nothing
		// TODO: remove check "d >= 0" after go 1.8 is no longer supported
		time.Sleep(r.minPeriod - d)
	}
	r.lastErrorTime = time.Now()
}

// GetCaller returns the caller of the function that calls it.
func GetCaller() string {
	var pc [1]uintptr
	runtime.Callers(3, pc[:])
	f := runtime.FuncForPC(pc[0])
	if f == nil {
		return fmt.Sprintf("Unable to find caller")
	}
	return f.Name()
}

// RecoverFromPanic replaces the specified error with an error containing the
// original error, and  the call tree when a panic occurs. This enables error
// handlers to handle errors and panics the same way.
func RecoverFromPanic(err *error) {
	if r := recover(); r != nil {
		callers := getCallers()

		*err = fmt.Errorf(
			"recovered from panic %q. (err=%v) Call stack:\n%v",
			r,
			*err,
			callers)
	}
}

func newDedupingErrorHandler(cacheSize, errorDepth int, delta time.Duration, similar float64) *dedupingErrorHandler {
	d := &dedupingErrorHandler{
		cache: lru.New(cacheSize),
		count: make(map[errKey]errVal),

		errorDepth: errorDepth,
		delta:      delta,
		Similar:    similar,
		clock:      clock.RealClock{},
	}

	d.cache.OnEvicted = func(key lru.Key, _ interface{}) {
		// remove the associated entry in the count map when this key is evicted from the LRU cache
		// d.cache.Add is only invoked when the mutex is held, so this delete is not a data race
		delete(d.count, key.(errKey))
	}
	d.logErrorHandler = d.logError
	d.getStackHandler = d.getStack

	return d
}

// dedupingErrorHandler provides a go routine safe error handler via handleErr.
// It tracks errors via the caller stack, the error type and the error message (err.Error() value).
// An error is considered unique based on these properties (see errKey), with one exception:
// errors with the same stack and type are considered equal if their message is similar enough (see getVal).
// To prevent from using an infinite amount of memory, it uses a LRU cache to purge old error values.
// The cache and count map are separated to allow easy access to all keys, which is required for fuzzy matching.
type dedupingErrorHandler struct {
	mutex sync.Mutex

	// cache tracks (stack + type + message) and cleans up old entries in count as they roll off the cache
	cache *lru.Cache
	// count tracks (stack + type + message) -> (count + logged)
	// since rudimentaryErrorBackoff rate limits HandleError to 1000 errors/second,
	// this counter will effectively never overflow (nothing bad happens even if it does)
	count map[errKey]errVal

	// errorDepth is how many frames to skip from handleErr
	errorDepth int

	// delta is the minimum difference between the current time and
	// errVal.logged required for us to log the associated error again
	delta time.Duration

	// Similar is the levenshtein ratio used to determine equivalence
	// TODO unexport this field once we change the default value from 1
	Similar float64

	// clock allows us to control time in unit tests
	clock clock.Clock

	// logErrorHandler allows us to check err and count in unit tests
	logErrorHandler func(err error, count uint64)

	// getStackHandler allows us to control the perceived stack in unit tests
	getStackHandler func() (stack string)
}

// errKey tracks uniqueness based on the caller's stack and the type/message of the error
// it is stored in both the count map and the LRU cache
// it is removed from the count map when it gets evicted from the LRU cache
// it is comparable via ==
type errKey struct {
	stack   string
	errType reflect.Type
	// message is the err.Error() value
	message string
}

// errVal tracks how many times we have seen an error, and the last time we logged it
type errVal struct {
	count  uint64
	logged time.Time
}

// handleErr logs the given error if it is considered new or "not recently logged"
// currently it logs errors whenever:
// 1. the associated errKey does not exist in d.count (see d.getVal for specifics on how this is calculated)
// 2. the associated counter is a power of two
// 3. the associated logged time's delta from the current time is greater than d.delta
func (d *dedupingErrorHandler) handleErr(err error) {
	// the operations below that do not acquire the lock do not mutate d

	// we must determine our stack in this function since getStack counts frames
	stack := d.getStackHandler()
	key := errKey{stack: stack, errType: reflect.TypeOf(err), message: err.Error()}

	// operations after this point can mutate d
	// we must acquire the lock before calling getVal since we cannot
	// have concurrent reads and writes against d.count
	d.mutex.Lock()
	defer d.mutex.Unlock()

	// increment our counter
	val, isNewErr := d.getVal(&key)
	val.count++

	if isNewErr {
		// we did not find the error, so add
		// the associated entry in the LRU cache
		d.cache.Add(key, nil)
	} else {
		// we found this error, so tell the cache that we saw it
		// cache.Get should always return nil, true
		d.cache.Get(key)
	}

	// determine if we need to log this time
	if isNewErr || isPowerOfTwo(val.count) || d.clock.Since(val.logged) >= d.delta {
		val.logged = d.clock.Now()
		d.logErrorHandler(err, val.count)
	}

	// update the counter in the map after we determine if we need to log it
	d.count[key] = val
}

// getVal returns the errVal associated with key, and if the key represents a new error.
// If the exact key does not exist in d.count, a similar enough fuzzy match on key.message
// will be used as a fallback (the stack and error type must always match via ==).
// If fuzzy matching is used, the given key's message will be updated to the similar message.
// Thus this method requires that key be passed in as a pointer.
func (d *dedupingErrorHandler) getVal(key *errKey) (errVal, bool) {
	val, isOldErr := d.count[*key]
	// found direct match, use that before doing levenshtein fuzzy lookup
	if isOldErr {
		return val, false // return false because this is not a new error
	}

	// Do not bother doing fuzzy checks if the direct match failed and we are set to require identical strings
	// TODO remove this once we have a useful default value for similar
	if d.Similar >= 1 {
		return errVal{}, true // return true because this is a new error
	}

	var (
		// fuzzySimilarity is initialized to d.similar since that is the
		// lowest similarity value we consider as the strings being "equal"
		fuzzySimilarity = d.Similar
		fuzzyFound      bool
		fuzzyMessage    string
		fuzzyVal        errVal
	)

	// we have to iterate over the whole map to do fuzzy matching on errKey.message
	// this is ok because d.count should always be relatively small
	// note that we cannot exit this loop early since we want to pick the most similar message
	for k, v := range d.count {
		// the stack and error type must match
		if key.stack == k.stack && key.errType == k.errType {
			// now we perform fuzzy matching on the message
			if s := levenshtein.Similarity(key.message, k.message, nil); s > fuzzySimilarity {
				fuzzySimilarity = s
				fuzzyFound = true
				fuzzyMessage = k.message
				fuzzyVal = v
			}
		}
	}

	if fuzzyFound {
		// update the input key's message to match our fuzzy search
		key.message = fuzzyMessage
		return fuzzyVal, false // return false because we do not consider this a new error
	}

	return errVal{}, true // return true because this is a new error
}

// logError uses glog to log at the call site of HandleError
// it must be called from dedupingErrorHandler.handleErr
func (d *dedupingErrorHandler) logError(err error, count uint64) {
	// get the full name of the file since glog likes to strip it
	// TODO if we are ok with multi line messages, we could simply print the whole stack
	_, file, line, ok := runtime.Caller(d.errorDepth)
	if !ok {
		file = "unknown_file"
		line = -1
	}
	glog.ErrorDepth(d.errorDepth, fmt.Sprintf("%s:%d err=%v count=%d value=%#v", file, line, err, count, err))
}

var (
	hexNumberRE  = regexp.MustCompile(`0x[0-9a-f]+`)
	emptyAddress = []byte("0x?")
)

// getStack returns the important part of the stack trace
// it must be called from dedupingErrorHandler.handleErr
// it must not mutate d (or even read d.cache/d.count) since it is called when the lock is not held
func (d *dedupingErrorHandler) getStack() string {
	// remove all hex addresses from the stack dump because closures can have volatile values
	stack := string(hexNumberRE.ReplaceAll(debug.Stack(), emptyAddress))
	// strip the redundant stuff at the top of the stack
	// add 1 to error depth for debug.Stack (since it calls runtime.Stack), times the sum by 2 since each frame has 2 lines
	// add 1 for go routine number header
	strip := (d.errorDepth+1)*2 + 1
	stackLines := strings.Split(stack, "\n")
	// do not panic trying to strip a stack trace that does not meet our expectations
	if strip >= len(stackLines) {
		return stack
	}
	return strings.Join(stackLines[strip:], "\n")
}

func isPowerOfTwo(n uint64) bool {
	return (n & (n - 1)) == 0
}
