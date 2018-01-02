package stressClient

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"time"
)

// ###############################################
// A selection of methods to manage the write path
// ###############################################

// Packages up Package from channel in goroutine
func (sc *stressClient) spinOffWritePackage(p Package, serv int) {
	sc.Add(1)
	sc.wc.Increment()
	go func() {
		sc.retry(p, time.Duration(time.Nanosecond), serv)
		sc.Done()
		sc.wc.Decrement()
	}()
}

// Implements backoff and retry logic for 500 responses
func (sc *stressClient) retry(p Package, backoff time.Duration, serv int) {

	// Set Backoff Interval to 500ms
	backoffInterval := time.Duration(500 * time.Millisecond)

	// Arithmetic backoff for kicks
	bo := backoff + backoffInterval

	// Make the write request
	resp, elapsed, err := sc.prepareWrite(p.Body, serv)

	// Find number of times request has been retried
	numBackoffs := int(bo/backoffInterval) - 1

	// On 500 responses, resp == nil. This logic keeps program for panicing
	var statusCode int

	if resp == nil {
		statusCode = 500
	} else {
		statusCode = resp.StatusCode
	}

	// Make a point for reporting
	point := sc.writePoint(numBackoffs, p.StatementID, statusCode, elapsed, p.Tracer.Tags, len(p.Body))

	// Send the Response(point, tracer)
	sc.responseChan <- NewResponse(point, p.Tracer)

	// BatchInterval enforcement
	bi, _ := time.ParseDuration(sc.wdelay)
	time.Sleep(bi)

	// Retry if the statusCode was not 204 or the err != nil
	if !(statusCode == 204) || err != nil {
		// Increment the *Tracer waitgroup if we are going to retry the request
		p.Tracer.Add(1)
		// Log the error if there is one
		fmt.Println(err)
		// Backoff enforcement
		time.Sleep(bo)
		sc.retry(p, bo, serv)
	}

}

// Prepares to send the POST request
func (sc *stressClient) prepareWrite(points []byte, serv int) (*http.Response, time.Duration, error) {

	// Construct address string
	var writeTemplate string
	if sc.ssl {
		writeTemplate = "https://%v/write?db=%v&precision=%v&u=%v&p=%v"
	} else {
		writeTemplate = "http://%v/write?db=%v&precision=%v&u=%v&p=%v"
	}
	address := fmt.Sprintf(writeTemplate, sc.addresses[serv], sc.database, sc.precision, sc.username, sc.password)

	// Start timer
	t := time.Now()
	resp, err := makePost(address, bytes.NewBuffer(points))
	elapsed := time.Since(t)

	return resp, elapsed, err
}

// Send POST request, read it, and handle errors
func makePost(url string, points io.Reader) (*http.Response, error) {

	resp, err := http.Post(url, "text/plain", points)

	if err != nil {
		return resp, fmt.Errorf("Error making write POST request\n  error: %v\n  url: %v\n", err, url)
	}

	body, _ := ioutil.ReadAll(resp.Body)

	if resp.StatusCode != 204 {
		return resp, fmt.Errorf("Write returned non-204 status code\n  StatusCode: %v\n  InfluxDB Error: %v\n", resp.StatusCode, string(body))
	}

	resp.Body.Close()

	return resp, nil
}
