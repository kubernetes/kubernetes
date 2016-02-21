package metrics

import (
	"encoding/json"
	"fmt"
	// "time"
)

// MetricType restrictions
type MetricType int

const (
	Gauge = iota
	Availability
	Counter
	Generic
)

var longForm = []string{
	"gauges",
	"availability",
	"counters",
	"metrics",
}

var shortForm = []string{
	"gauge",
	"availability",
	"counter",
	"metrics",
}

func (self MetricType) validate() error {
	if int(self) > len(longForm) && int(self) > len(shortForm) {
		return fmt.Errorf("Given MetricType value %d is not valid", self)
	}
	return nil
}

func (self MetricType) String() string {
	if err := self.validate(); err != nil {
		return "unknown"
	}
	return longForm[self]
}

func (self MetricType) shortForm() string {
	if err := self.validate(); err != nil {
		return "unknown"
	}
	return shortForm[self]
}

// Custom unmarshaller
func (self *MetricType) UnmarshalJSON(b []byte) error {
	var f interface{}
	err := json.Unmarshal(b, &f)
	if err != nil {
		return err
	}

	if str, ok := f.(string); ok {
		for i, v := range shortForm {
			if str == v {
				*self = MetricType(i)
				break
			}
		}
	}

	return nil
}

func (self MetricType) MarshalJSON() ([]byte, error) {
	return json.Marshal(self.String())
}

type SortKey struct {
	Tenant string
	Type   MetricType
}

// Hawkular-Metrics external structs
// Do I need external.. hmph.

type MetricHeader struct {
	Tenant string      `json:"-"`
	Type   MetricType  `json:"-"`
	Id     string      `json:"id"`
	Data   []Datapoint `json:"data"`
}

// Value should be convertible to float64 for numeric values
// Timestamp is milliseconds since epoch
type Datapoint struct {
	Timestamp int64             `json:"timestamp"`
	Value     interface{}       `json:"value"`
	Tags      map[string]string `json:"tags,omitempty"`
}

type HawkularError struct {
	ErrorMsg string `json:"errorMsg"`
}

type MetricDefinition struct {
	Tenant        string            `json:"-"`
	Type          MetricType        `json:"type,omitempty"`
	Id            string            `json:"id"`
	Tags          map[string]string `json:"tags,omitempty"`
	RetentionTime int               `json:"dataRetention,omitempty"`
}

// TODO Fix the Start & End to return a time.Time
type Bucketpoint struct {
	Start       int64        `json:"start"`
	End         int64        `json:"end"`
	Min         float64      `json:"min"`
	Max         float64      `json:"max"`
	Avg         float64      `json:"avg"`
	Median      float64      `json:"median"`
	Empty       bool         `json:"empty"`
	Samples     int64        `json:"samples"`
	Percentiles []Percentile `json:"percentiles"`
}

type Percentile struct {
	Quantile float64 `json:"quantile"`
	Value    float64 `json:"value"`
}
