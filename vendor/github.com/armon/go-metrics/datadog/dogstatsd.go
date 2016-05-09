package datadog

import (
	"fmt"
	"strings"

	"github.com/DataDog/datadog-go/statsd"
)

// DogStatsdSink provides a MetricSink that can be used
// with a dogstatsd server. It utilizes the Dogstatsd client at github.com/DataDog/datadog-go/statsd
type DogStatsdSink struct {
	client            *statsd.Client
	hostName          string
	propagateHostname bool
}

// NewDogStatsdSink is used to create a new DogStatsdSink with sane defaults
func NewDogStatsdSink(addr string, hostName string) (*DogStatsdSink, error) {
	client, err := statsd.New(addr)
	if err != nil {
		return nil, err
	}
	sink := &DogStatsdSink{
		client:            client,
		hostName:          hostName,
		propagateHostname: false,
	}
	return sink, nil
}

// SetTags sets common tags on the Dogstatsd Client that will be sent
// along with all dogstatsd packets.
// Ref: http://docs.datadoghq.com/guides/dogstatsd/#tags
func (s *DogStatsdSink) SetTags(tags []string) {
	s.client.Tags = tags
}

// EnableHostnamePropagation forces a Dogstatsd `host` tag with the value specified by `s.HostName`
// Since the go-metrics package has its own mechanism for attaching a hostname to metrics,
// setting the `propagateHostname` flag ensures that `s.HostName` overrides the host tag naively set by the DogStatsd server
func (s *DogStatsdSink) EnableHostNamePropagation() {
	s.propagateHostname = true
}

func (s *DogStatsdSink) flattenKey(parts []string) string {
	joined := strings.Join(parts, ".")
	return strings.Map(func(r rune) rune {
		switch r {
		case ':':
			fallthrough
		case ' ':
			return '_'
		default:
			return r
		}
	}, joined)
}

func (s *DogStatsdSink) parseKey(key []string) ([]string, []string) {
	// Since DogStatsd supports dimensionality via tags on metric keys, this sink's approach is to splice the hostname out of the key in favor of a `host` tag
	// The `host` tag is either forced here, or set downstream by the DogStatsd server

	var tags []string
	hostName := s.hostName

	//Splice the hostname out of the key
	for i, el := range key {
		if el == hostName {
			key = append(key[:i], key[i+1:]...)
		}
	}

	if s.propagateHostname {
		tags = append(tags, fmt.Sprintf("host:%s", hostName))
	}
	return key, tags
}

// Implementation of methods in the MetricSink interface

func (s *DogStatsdSink) SetGauge(key []string, val float32) {
	s.SetGaugeWithTags(key, val, []string{})
}

func (s *DogStatsdSink) IncrCounter(key []string, val float32) {
	s.IncrCounterWithTags(key, val, []string{})
}

// EmitKey is not implemented since DogStatsd does not provide a metric type that holds an
// arbitrary number of values
func (s *DogStatsdSink) EmitKey(key []string, val float32) {
}

func (s *DogStatsdSink) AddSample(key []string, val float32) {
	s.AddSampleWithTags(key, val, []string{})
}

// The following ...WithTags methods correspond to Datadog's Tag extension to Statsd.
// http://docs.datadoghq.com/guides/dogstatsd/#tags

func (s *DogStatsdSink) SetGaugeWithTags(key []string, val float32, tags []string) {
	flatKey, tags := s.getFlatkeyAndCombinedTags(key, tags)
	rate := 1.0
	s.client.Gauge(flatKey, float64(val), tags, rate)
}

func (s *DogStatsdSink) IncrCounterWithTags(key []string, val float32, tags []string) {
	flatKey, tags := s.getFlatkeyAndCombinedTags(key, tags)
	rate := 1.0
	s.client.Count(flatKey, int64(val), tags, rate)
}

func (s *DogStatsdSink) AddSampleWithTags(key []string, val float32, tags []string) {
	flatKey, tags := s.getFlatkeyAndCombinedTags(key, tags)
	rate := 1.0
	s.client.TimeInMilliseconds(flatKey, float64(val), tags, rate)
}

func (s *DogStatsdSink) getFlatkeyAndCombinedTags(key []string, tags []string) (flattenedKey string, combinedTags []string) {
	key, hostTags := s.parseKey(key)
	flatKey := s.flattenKey(key)
	tags = append(tags, hostTags...)
	return flatKey, tags
}
