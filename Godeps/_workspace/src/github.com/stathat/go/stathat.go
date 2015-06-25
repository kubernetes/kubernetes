// Copyright (C) 2012 Numerotron Inc.
// Use of this source code is governed by an MIT-style license
// that can be found in the LICENSE file.

// Copyright 2012 Numerotron Inc.
// Use of this source code is governed by an MIT-style license
// that can be found in the LICENSE file.
//
// Developed at www.stathat.com by Patrick Crosby
// Contact us on twitter with any questions:  twitter.com/stat_hat

// The stathat package makes it easy to post any values to your StatHat
// account.
package stathat

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"
)

const hostname = "api.stathat.com"

type statKind int

const (
	_                 = iota
	kcounter statKind = iota
	kvalue
)

func (sk statKind) classicPath() string {
	switch sk {
	case kcounter:
		return "/c"
	case kvalue:
		return "/v"
	}
	return ""
}

type apiKind int

const (
	_               = iota
	classic apiKind = iota
	ez
)

func (ak apiKind) path(sk statKind) string {
	switch ak {
	case ez:
		return "/ez"
	case classic:
		return sk.classicPath()
	}
	return ""
}

type statReport struct {
	StatKey   string
	UserKey   string
	Value     float64
	Timestamp int64
	statType  statKind
	apiType   apiKind
}

// Reporter is a StatHat client that can report stat values/counts to the servers.
type Reporter struct {
	reports chan *statReport
	done    chan bool
	client  *http.Client
	wg      *sync.WaitGroup
}

// NewReporter returns a new Reporter.  You must specify the channel bufferSize and the
// goroutine poolSize.  You can pass in nil for the transport and it will use the
// default http transport.
func NewReporter(bufferSize, poolSize int, transport http.RoundTripper) *Reporter {
	r := new(Reporter)
	r.client = &http.Client{Transport: transport}
	r.reports = make(chan *statReport, bufferSize)
	r.done = make(chan bool)
	r.wg = new(sync.WaitGroup)
	for i := 0; i < poolSize; i++ {
		r.wg.Add(1)
		go r.processReports()
	}
	return r
}

// DefaultReporter is the default instance of *Reporter.
var DefaultReporter = NewReporter(100000, 10, nil)

var testingEnv = false

type testPost struct {
	url    string
	values url.Values
}

var testPostChannel chan *testPost

// The Verbose flag determines if the package should write verbose output to stdout.
var Verbose = false

func setTesting() {
	testingEnv = true
	testPostChannel = make(chan *testPost)
}

func newEZStatCount(statName, ezkey string, count int) *statReport {
	return &statReport{StatKey: statName,
		UserKey:  ezkey,
		Value:    float64(count),
		statType: kcounter,
		apiType:  ez}
}

func newEZStatValue(statName, ezkey string, value float64) *statReport {
	return &statReport{StatKey: statName,
		UserKey:  ezkey,
		Value:    value,
		statType: kvalue,
		apiType:  ez}
}

func newClassicStatCount(statKey, userKey string, count int) *statReport {
	return &statReport{StatKey: statKey,
		UserKey:  userKey,
		Value:    float64(count),
		statType: kcounter,
		apiType:  classic}
}

func newClassicStatValue(statKey, userKey string, value float64) *statReport {
	return &statReport{StatKey: statKey,
		UserKey:  userKey,
		Value:    value,
		statType: kvalue,
		apiType:  classic}
}

func (sr *statReport) values() url.Values {
	switch sr.apiType {
	case ez:
		return sr.ezValues()
	case classic:
		return sr.classicValues()
	}

	return nil
}

func (sr *statReport) ezValues() url.Values {
	switch sr.statType {
	case kcounter:
		return sr.ezCounterValues()
	case kvalue:
		return sr.ezValueValues()
	}
	return nil
}

func (sr *statReport) classicValues() url.Values {
	switch sr.statType {
	case kcounter:
		return sr.classicCounterValues()
	case kvalue:
		return sr.classicValueValues()
	}
	return nil
}

func (sr *statReport) ezCommonValues() url.Values {
	result := make(url.Values)
	result.Set("stat", sr.StatKey)
	result.Set("ezkey", sr.UserKey)
	if sr.Timestamp > 0 {
		result.Set("t", sr.timeString())
	}
	return result
}

func (sr *statReport) classicCommonValues() url.Values {
	result := make(url.Values)
	result.Set("key", sr.StatKey)
	result.Set("ukey", sr.UserKey)
	if sr.Timestamp > 0 {
		result.Set("t", sr.timeString())
	}
	return result
}

func (sr *statReport) ezCounterValues() url.Values {
	result := sr.ezCommonValues()
	result.Set("count", sr.valueString())
	return result
}

func (sr *statReport) ezValueValues() url.Values {
	result := sr.ezCommonValues()
	result.Set("value", sr.valueString())
	return result
}

func (sr *statReport) classicCounterValues() url.Values {
	result := sr.classicCommonValues()
	result.Set("count", sr.valueString())
	return result
}

func (sr *statReport) classicValueValues() url.Values {
	result := sr.classicCommonValues()
	result.Set("value", sr.valueString())
	return result
}

func (sr *statReport) valueString() string {
	return strconv.FormatFloat(sr.Value, 'g', -1, 64)
}

func (sr *statReport) timeString() string {
	return strconv.FormatInt(sr.Timestamp, 10)
}

func (sr *statReport) path() string {
	return sr.apiType.path(sr.statType)
}

func (sr *statReport) url() string {
	return fmt.Sprintf("http://%s%s", hostname, sr.path())
}

// Using the classic API, posts a count to a stat using DefaultReporter.
func PostCount(statKey, userKey string, count int) error {
	return DefaultReporter.PostCount(statKey, userKey, count)
}

// Using the classic API, posts a count to a stat using DefaultReporter at a specific
// time.
func PostCountTime(statKey, userKey string, count int, timestamp int64) error {
	return DefaultReporter.PostCountTime(statKey, userKey, count, timestamp)
}

// Using the classic API, posts a count of 1 to a stat using DefaultReporter.
func PostCountOne(statKey, userKey string) error {
	return DefaultReporter.PostCountOne(statKey, userKey)
}

// Using the classic API, posts a value to a stat using DefaultReporter.
func PostValue(statKey, userKey string, value float64) error {
	return DefaultReporter.PostValue(statKey, userKey, value)
}

// Using the classic API, posts a value to a stat at a specific time using DefaultReporter.
func PostValueTime(statKey, userKey string, value float64, timestamp int64) error {
	return DefaultReporter.PostValueTime(statKey, userKey, value, timestamp)
}

// Using the EZ API, posts a count of 1 to a stat using DefaultReporter.
func PostEZCountOne(statName, ezkey string) error {
	return DefaultReporter.PostEZCountOne(statName, ezkey)
}

// Using the EZ API, posts a count to a stat using DefaultReporter.
func PostEZCount(statName, ezkey string, count int) error {
	return DefaultReporter.PostEZCount(statName, ezkey, count)
}

// Using the EZ API, posts a count to a stat at a specific time using DefaultReporter.
func PostEZCountTime(statName, ezkey string, count int, timestamp int64) error {
	return DefaultReporter.PostEZCountTime(statName, ezkey, count, timestamp)
}

// Using the EZ API, posts a value to a stat using DefaultReporter.
func PostEZValue(statName, ezkey string, value float64) error {
	return DefaultReporter.PostEZValue(statName, ezkey, value)
}

// Using the EZ API, posts a value to a stat at a specific time using DefaultReporter.
func PostEZValueTime(statName, ezkey string, value float64, timestamp int64) error {
	return DefaultReporter.PostEZValueTime(statName, ezkey, value, timestamp)
}

// Wait for all stats to be sent, or until timeout. Useful for simple command-
// line apps to defer a call to this in main()
func WaitUntilFinished(timeout time.Duration) bool {
	return DefaultReporter.WaitUntilFinished(timeout)
}

// Using the classic API, posts a count to a stat.
func (r *Reporter) PostCount(statKey, userKey string, count int) error {
	r.reports <- newClassicStatCount(statKey, userKey, count)
	return nil
}

// Using the classic API, posts a count to a stat at a specific time.
func (r *Reporter) PostCountTime(statKey, userKey string, count int, timestamp int64) error {
	x := newClassicStatCount(statKey, userKey, count)
	x.Timestamp = timestamp
	r.reports <- x
	return nil
}

// Using the classic API, posts a count of 1 to a stat.
func (r *Reporter) PostCountOne(statKey, userKey string) error {
	return r.PostCount(statKey, userKey, 1)
}

// Using the classic API, posts a value to a stat.
func (r *Reporter) PostValue(statKey, userKey string, value float64) error {
	r.reports <- newClassicStatValue(statKey, userKey, value)
	return nil
}

// Using the classic API, posts a value to a stat at a specific time.
func (r *Reporter) PostValueTime(statKey, userKey string, value float64, timestamp int64) error {
	x := newClassicStatValue(statKey, userKey, value)
	x.Timestamp = timestamp
	r.reports <- x
	return nil
}

// Using the EZ API, posts a count of 1 to a stat.
func (r *Reporter) PostEZCountOne(statName, ezkey string) error {
	return r.PostEZCount(statName, ezkey, 1)
}

// Using the EZ API, posts a count to a stat.
func (r *Reporter) PostEZCount(statName, ezkey string, count int) error {
	r.reports <- newEZStatCount(statName, ezkey, count)
	return nil
}

// Using the EZ API, posts a count to a stat at a specific time.
func (r *Reporter) PostEZCountTime(statName, ezkey string, count int, timestamp int64) error {
	x := newEZStatCount(statName, ezkey, count)
	x.Timestamp = timestamp
	r.reports <- x
	return nil
}

// Using the EZ API, posts a value to a stat.
func (r *Reporter) PostEZValue(statName, ezkey string, value float64) error {
	r.reports <- newEZStatValue(statName, ezkey, value)
	return nil
}

// Using the EZ API, posts a value to a stat at a specific time.
func (r *Reporter) PostEZValueTime(statName, ezkey string, value float64, timestamp int64) error {
	x := newEZStatValue(statName, ezkey, value)
	x.Timestamp = timestamp
	r.reports <- x
	return nil
}

func (r *Reporter) processReports() {
	for {
		sr, ok := <-r.reports

		if !ok {
			if Verbose {
				log.Printf("channel closed, stopping processReports()")
			}
			break
		}

		if Verbose {
			log.Printf("posting stat to stathat: %s, %v", sr.url(), sr.values())
		}

		if testingEnv {
			if Verbose {
				log.Printf("in test mode, putting stat on testPostChannel")
			}
			testPostChannel <- &testPost{sr.url(), sr.values()}
			continue
		}

		resp, err := r.client.PostForm(sr.url(), sr.values())
		if err != nil {
			log.Printf("error posting stat to stathat: %s", err)
			continue
		}

		if Verbose {
			body, _ := ioutil.ReadAll(resp.Body)
			log.Printf("stathat post result: %s", body)
		}

		resp.Body.Close()
	}
	r.wg.Done()
}

func (r *Reporter) finish() {
	close(r.reports)
	r.wg.Wait()
	r.done <- true
}

// Wait for all stats to be sent, or until timeout. Useful for simple command-
// line apps to defer a call to this in main()
func (r *Reporter) WaitUntilFinished(timeout time.Duration) bool {
	go r.finish()
	select {
	case <-r.done:
		return true
	case <-time.After(timeout):
		return false
	}
	return false
}
