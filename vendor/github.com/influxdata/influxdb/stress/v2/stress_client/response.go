package stressClient

import (
	"log"

	influx "github.com/influxdata/influxdb/client/v2"
)

// Response holds data scraped from InfluxDB HTTP responses turned into a *influx.Point for reporting
// See reporting.go for more information
// The Tracer contains a wait group sent from the statement. It needs to be decremented when the Response is consumed
type Response struct {
	Point  *influx.Point
	Tracer *Tracer
}

// NewResponse creates a new instance of Response
func NewResponse(pt *influx.Point, tr *Tracer) Response {
	return Response{
		Point:  pt,
		Tracer: tr,
	}
}

// AddTags adds additional tags to the point held in Response and returns the point
func (resp Response) AddTags(newTags map[string]string) *influx.Point {

	// Pull off the current tags
	tags := resp.Point.Tags()

	// Add the new tags to the current tags
	for tag, tagValue := range newTags {
		tags[tag] = tagValue
	}

	// Make a new point
	pt, err := influx.NewPoint(resp.Point.Name(), tags, resp.Point.Fields(), resp.Point.Time())

	// panic on error
	if err != nil {
		log.Fatalf("Error adding tags to response point\n  point: %v\n  tags:%v\n  error: %v\n", resp.Point, newTags, err)
	}

	return pt
}
