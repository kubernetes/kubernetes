# Circonus metrics tracking for Go applications

This library supports named counters, gauges and histograms.
It also provides convenience wrappers for registering latency
instrumented functions with Go's builtin http server.

Initializing only requires setting an ApiToken.

## Example

**rough and simple**

```go
package main

import (
	"log"
	"math/rand"
	"os"
	"time"

	cgm "github.com/circonus-labs/circonus-gometrics"
)

func main() {

	log.Println("Configuring cgm")

	cmc := &cgm.Config{}

	// Interval at which metrics are submitted to Circonus, default: 10 seconds
	cmc.Interval = "10s" // 10 seconds
	// Enable debug messages, default: false
	cmc.Debug = false
	// Send debug messages to specific log.Logger instance
	// default: if debug stderr, else, discard
	//cmc.CheckManager.Log = ...

	// Circonus API configuration options
	//
	// Token, no default (blank disables check manager)
	cmc.CheckManager.API.TokenKey = os.Getenv("CIRCONUS_API_TOKEN")
	// App name, default: circonus-gometrics
	cmc.CheckManager.API.TokenApp = os.Getenv("CIRCONUS_API_APP")
	// URL, default: https://api.circonus.com/v2
	cmc.CheckManager.API.URL = os.Getenv("CIRCONUS_API_URL")

	// Check configuration options
	//
	// precedence 1 - explicit submission_url
	// precedence 2 - specific check id (note: not a check bundle id)
	// precedence 3 - search using instanceId and searchTag
	// otherwise: if an applicable check is NOT specified or found, an
	//            attempt will be made to automatically create one
	//
	// Pre-existing httptrap check submission_url
	cmc.CheckManager.Check.SubmissionURL = os.Getenv("CIRCONUS_SUBMISION_URL")
	// Pre-existing httptrap check id (check not check bundle)
	cmc.CheckManager.Check.ID = ""
	// if neither a submission url nor check id are provided, an attempt will be made to find an existing
	// httptrap check by using the circonus api to search for a check matching the following criteria:
	//      an active check,
	//      of type httptrap,
	//      where the target/host is equal to InstanceId - see below
	//      and the check has a tag equal to SearchTag - see below
	// Instance ID - an identifier for the 'group of metrics emitted by this process or service'
	//               this is used as the value for check.target (aka host)
	// default: 'hostname':'program name'
	// note: for a persistent instance that is ephemeral or transient where metric continuity is
	//       desired set this explicitly so that the current hostname will not be used.
	cmc.CheckManager.Check.InstanceID = ""
	// Search tag - a specific tag which, when coupled with the instanceId serves to identify the
	// origin and/or grouping of the metrics
	// default: service:application name (e.g. service:consul)
	cmc.CheckManager.Check.SearchTag = ""
	// Check secret, default: generated when a check needs to be created
	cmc.CheckManager.Check.Secret = ""
	// Check tags, array of strings, additional tags to add to a new check, default: none
	//cmc.CheckManager.Check.Tags = []string{"category:tagname"}
	// max amount of time to to hold on to a submission url
	// when a given submission fails (due to retries) if the
	// time the url was last updated is > than this, the trap
	// url will be refreshed (e.g. if the broker is changed
	// in the UI) default 5 minutes
	cmc.CheckManager.Check.MaxURLAge = "5m"
	// custom display name for check, default: "InstanceId /cgm"
	cmc.CheckManager.Check.DisplayName = ""
    // force metric activation - if a metric has been disabled via the UI
	// the default behavior is to *not* re-activate the metric; this setting
	// overrides the behavior and will re-activate the metric when it is
	// encountered. "(true|false)", default "false"
	cmc.CheckManager.Check.ForceMetricActivation = "false"

	// Broker configuration options
	//
	// Broker ID of specific broker to use, default: random enterprise broker or
	// Circonus default if no enterprise brokers are available.
	// default: only used if set
	cmc.CheckManager.Broker.ID = ""
	// used to select a broker with the same tag (e.g. can be used to dictate that a broker
	// serving a specific location should be used. "dc:sfo", "location:new_york", "zone:us-west")
	// if more than one broker has the tag, one will be selected randomly from the resulting list
	// default: not used unless != ""
	cmc.CheckManager.Broker.SelectTag = ""
	// longest time to wait for a broker connection (if latency is > the broker will
	// be considered invalid and not available for selection.), default: 500 milliseconds
	cmc.CheckManager.Broker.MaxResponseTime = "500ms"
	// if broker Id or SelectTag are not specified, a broker will be selected randomly
	// from the list of brokers available to the api token. enterprise brokers take precedence
	// viable brokers are "active", have the "httptrap" module enabled, are reachable and respond
	// within MaxResponseTime.

	log.Println("Creating new cgm instance")

	metrics, err := cgm.NewCirconusMetrics(cmc)
	if err != nil {
		panic(err)
	}

	src := rand.NewSource(time.Now().UnixNano())
	rnd := rand.New(src)

	log.Println("Starting cgm internal auto-flush timer")
	metrics.Start()

	log.Println("Starting to send metrics")

	// number of "sets" of metrics to send (a minute worth)
	max := 60

	for i := 1; i < max; i++ {
		log.Printf("\tmetric set %d of %d", i, 60)
		metrics.Timing("ding", rnd.Float64()*10)
		metrics.Increment("dong")
		metrics.Gauge("dang", 10)
		time.Sleep(1000 * time.Millisecond)
	}

	log.Println("Flushing any outstanding metrics manually")
	metrics.Flush()

}
```

### HTTP Handler wrapping

```
http.HandleFunc("/", metrics.TrackHTTPLatency("/", handler_func))
```

### HTTP latency example

```
package main

import (
    "os"
    "fmt"
    "net/http"
    cgm "github.com/circonus-labs/circonus-gometrics"
)

func main() {
    cmc := &cgm.Config{}
    cmc.CheckManager.API.TokenKey = os.Getenv("CIRCONUS_API_TOKEN")

    metrics, err := cgm.NewCirconusMetrics(cmc)
    if err != nil {
        panic(err)
    }
    metrics.Start()

    http.HandleFunc("/", metrics.TrackHTTPLatency("/", func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintf(w, "Hello, %s!", r.URL.Path[1:])
    }))
    http.ListenAndServe(":8080", http.DefaultServeMux)
}

```

Unless otherwise noted, the source files are distributed under the BSD-style license found in the LICENSE file.
