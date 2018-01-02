package stressClient

import (
	"strings"
	"sync"
)

// Type refers to the different Package types
type Type int

// There are two package types, Write and Query
const (
	Write Type = iota
	Query
)

func startStressClient(packageCh <-chan Package, directiveCh <-chan Directive, responseCh chan<- Response, testID string) {

	c := &stressClient{
		testID: testID,

		addresses: []string{"localhost:8086"},
		ssl:       false,
		username:  "",
		password:  "",
		precision: "ns",
		database:  "stress",
		startDate: "2016-01-01",
		qdelay:    "0s",
		wdelay:    "0s",

		wconc: 10,
		qconc: 5,

		packageChan:   packageCh,
		directiveChan: directiveCh,

		responseChan: responseCh,
	}
	// start listening for writes and queries
	go c.listen()
	// start listening for state changes
	go c.directiveListen()
}

type stressClient struct {
	testID string

	// State for the Stress Test
	addresses []string
	precision string
	startDate string
	database  string
	wdelay    string
	qdelay    string
	username  string
	password  string
	ssl       bool

	// Channels from statements
	packageChan   <-chan Package
	directiveChan <-chan Directive

	// Response channel
	responseChan chan<- Response

	// Concurrency utilities
	sync.WaitGroup
	sync.Mutex

	// Concurrency Limit for Writes and Reads
	wconc int
	qconc int

	// Manage Read and Write concurrency seperately
	wc *ConcurrencyLimiter
	rc *ConcurrencyLimiter
}

// NewTestStressClient returns a blank stressClient for testing
func newTestStressClient(url string) (*stressClient, chan Directive, chan Package) {
	pkgChan := make(chan Package)
	dirChan := make(chan Directive)
	pe := &stressClient{
		testID:        "foo_id",
		addresses:     []string{url},
		precision:     "s",
		startDate:     "2016-01-01",
		database:      "fooDatabase",
		wdelay:        "50ms",
		qdelay:        "50ms",
		ssl:           false,
		username:      "",
		password:      "",
		wconc:         5,
		qconc:         5,
		packageChan:   pkgChan,
		directiveChan: dirChan,
		wc:            NewConcurrencyLimiter(1),
		rc:            NewConcurrencyLimiter(1),
	}
	return pe, dirChan, pkgChan
}

// stressClient starts listening for Packages on the main channel
func (sc *stressClient) listen() {
	defer sc.Wait()
	sc.wc = NewConcurrencyLimiter(sc.wconc)
	sc.rc = NewConcurrencyLimiter(sc.qconc)
	l := NewConcurrencyLimiter((sc.wconc + sc.qconc) * 2)
	counter := 0
	for p := range sc.packageChan {
		l.Increment()
		go func(p Package) {
			defer l.Decrement()
			switch p.T {
			case Write:
				sc.spinOffWritePackage(p, (counter % len(sc.addresses)))
			case Query:
				sc.spinOffQueryPackage(p, (counter % len(sc.addresses)))
			}
		}(p)
		counter++
	}

}

// Set handles all SET requests for test state
func (sc *stressClient) directiveListen() {
	for d := range sc.directiveChan {
		sc.Lock()
		switch d.Property {
		// addresses is a []string of target InfluxDB instance(s) for the test
		// comes in as a "|" seperated array of addresses
		case "addresses":
			addr := strings.Split(d.Value, "|")
			sc.addresses = addr
		// percison is the write precision for InfluxDB
		case "precision":
			sc.precision = d.Value
		// writeinterval is an optional delay between batches
		case "writeinterval":
			sc.wdelay = d.Value
		// queryinterval is an optional delay between the batches
		case "queryinterval":
			sc.qdelay = d.Value
		// database is the InfluxDB database to target for both writes and queries
		case "database":
			sc.database = d.Value
		// username for the target database
		case "username":
			sc.username = d.Value
		// username for the target database
		case "password":
			sc.password = d.Value
		// use https if sent true
		case "ssl":
			if d.Value == "true" {
				sc.ssl = true
			}
		// concurrency is the number concurrent writers to the database
		case "writeconcurrency":
			conc := parseInt(d.Value)
			sc.wconc = conc
			sc.wc.NewMax(conc)
		// concurrentqueries is the number of concurrent queriers database
		case "queryconcurrency":
			conc := parseInt(d.Value)
			sc.qconc = conc
			sc.rc.NewMax(conc)
		}
		d.Tracer.Done()
		sc.Unlock()
	}
}
