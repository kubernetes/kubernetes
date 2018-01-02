package stressClient

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"time"
)

func (sc *stressClient) spinOffQueryPackage(p Package, serv int) {
	sc.Add(1)
	sc.rc.Increment()
	go func() {
		// Send the query
		sc.prepareQuerySend(p, serv)
		sc.Done()
		sc.rc.Decrement()
	}()
}

// Prepares to send the GET request
func (sc *stressClient) prepareQuerySend(p Package, serv int) {

	var queryTemplate string
	if sc.ssl {
		queryTemplate = "https://%v/query?db=%v&q=%v&u=%v&p=%v"
	} else {
		queryTemplate = "http://%v/query?db=%v&q=%v&u=%v&p=%v"
	}
	queryURL := fmt.Sprintf(queryTemplate, sc.addresses[serv], sc.database, url.QueryEscape(string(p.Body)), sc.username, sc.password)

	// Send the query
	sc.makeGet(queryURL, p.StatementID, p.Tracer)

	// Query Interval enforcement
	qi, _ := time.ParseDuration(sc.qdelay)
	time.Sleep(qi)
}

// Sends the GET request, reads it, and handles errors
func (sc *stressClient) makeGet(addr, statementID string, tr *Tracer) {

	// Make GET request
	t := time.Now()
	resp, err := http.Get(addr)
	elapsed := time.Since(t)

	if err != nil {
		log.Printf("Error making Query HTTP request\n  error: %v\n", err)
	}

	defer resp.Body.Close()

	// Read body and return it for Reporting
	body, err := ioutil.ReadAll(resp.Body)

	if err != nil {
		log.Fatalf("Error reading Query response body\n  error: %v\n", err)
	}

	if resp.StatusCode != 200 {
		log.Printf("Query returned non 200 status\n  status: %v\n  error: %v\n", resp.StatusCode, string(body))
	}

	// Send the response
	sc.responseChan <- NewResponse(sc.queryPoint(statementID, body, resp.StatusCode, elapsed, tr.Tags), tr)
}

func success(r *http.Response) bool {
	// ADD success for tcp, udp, etc
	return r != nil && (r.StatusCode == 204 || r.StatusCode == 200)
}
