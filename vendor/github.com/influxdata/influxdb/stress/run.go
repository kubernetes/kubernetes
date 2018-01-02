package stress // import "github.com/influxdata/influxdb/stress"

import (
	"bytes"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// Point is an interface that is used to represent
// the abstract idea of a point in InfluxDB.
type Point interface {
	Line() []byte
	Graphite() []byte
	OpenJSON() []byte
	OpenTelnet() []byte
}

///////////////////////////////////////////////////
// Example Implementation of the Point Interface //
///////////////////////////////////////////////////

// KeyValue is an intermediate type that is used
// to express Tag and Field similarly.
type KeyValue struct {
	Key   string
	Value string
}

// Tag is a struct for a tag in influxdb.
type Tag KeyValue

// Field is a struct for a field in influxdb.
type Field KeyValue

// Tags is an slice of all the tags for a point.
type Tags []Tag

// Fields is an slice of all the fields for a point.
type Fields []Field

// tagset returns a byte array for a points tagset.
func (t Tags) tagset() []byte {
	var buf bytes.Buffer
	for _, tag := range t {
		buf.Write([]byte(fmt.Sprintf("%v=%v,", tag.Key, tag.Value)))
	}

	b := buf.Bytes()
	b = b[0 : len(b)-1]

	return b
}

// fieldset returns a byte array for a points fieldset.
func (f Fields) fieldset() []byte {
	var buf bytes.Buffer
	for _, field := range f {
		buf.Write([]byte(fmt.Sprintf("%v=%v,", field.Key, field.Value)))
	}

	b := buf.Bytes()
	b = b[0 : len(b)-1]

	return b
}

// StdPoint represents a point in InfluxDB
type StdPoint struct {
	Measurement string
	Tags        Tags
	Fields      Fields
	Timestamp   int64
}

// Line returns a byte array for a point in
// line-protocol format
func (p StdPoint) Line() []byte {
	var buf bytes.Buffer

	buf.Write([]byte(fmt.Sprintf("%v,", p.Measurement)))
	buf.Write(p.Tags.tagset())
	buf.Write([]byte(" "))
	buf.Write(p.Fields.fieldset())
	buf.Write([]byte(" "))
	buf.Write([]byte(fmt.Sprintf("%v", p.Timestamp)))

	byt := buf.Bytes()

	return byt
}

// Graphite returns a byte array for a point
// in graphite-protocol format
func (p StdPoint) Graphite() []byte {
	// TODO: implement
	// timestamp is at second level resolution
	// but can be specified as a float to get nanosecond
	// level precision
	t := "tag_1.tag_2.measurement[.field] acutal_value timestamp"
	return []byte(t)
}

// OpenJSON returns a byte array for a point
// in JSON format
func (p StdPoint) OpenJSON() []byte {
	// TODO: implement
	//[
	//    {
	//        "metric": "sys.cpu.nice",
	//        "timestamp": 1346846400,
	//        "value": 18,
	//        "tags": {
	//           "host": "web01",
	//           "dc": "lga"
	//        }
	//    },
	//    {
	//        "metric": "sys.cpu.nice",
	//        "timestamp": 1346846400,
	//        "value": 9,
	//        "tags": {
	//           "host": "web02",
	//           "dc": "lga"
	//        }
	//    }
	//]
	return []byte("hello")
}

// OpenTelnet returns a byte array for a point
// in OpenTSDB-telnet format
func (p StdPoint) OpenTelnet() []byte {
	// TODO: implement
	// timestamp can be 13 digits at most
	// sys.cpu.nice timestamp value tag_key_1=tag_value_1 tag_key_2=tag_value_2
	return []byte("hello")
}

////////////////////////////////////////

// response is the results making
// a request to influxdb.
type response struct {
	Resp  *http.Response
	Time  time.Time
	Timer *Timer
}

// Success returns true if the request
// was successful and false otherwise.
func (r response) Success() bool {
	// ADD success for tcp, udp, etc
	return !(r.Resp == nil || r.Resp.StatusCode != 204)
}

// WriteResponse is a response for a Writer
type WriteResponse response

// QueryResponse is a response for a Querier
type QueryResponse struct {
	response
	Body string
}

///////////////////////////////
// Definition of the Writer ///
///////////////////////////////

// PointGenerator is an interface for generating points.
type PointGenerator interface {
	Generate() (<-chan Point, error)
	Time() time.Time
}

// InfluxClient is an interface for writing data to the database.
type InfluxClient interface {
	Batch(ps <-chan Point, r chan<- response) error
	send(b []byte) (response, error)
	//ResponseHandler
}

// Writer is a PointGenerator and an InfluxClient.
type Writer struct {
	PointGenerator
	InfluxClient
}

// NewWriter returns a Writer.
func NewWriter(p PointGenerator, i InfluxClient) Writer {
	w := Writer{
		PointGenerator: p,
		InfluxClient:   i,
	}

	return w
}

////////////////////////////////
// Definition of the Querier ///
////////////////////////////////

// Query is query
type Query string

// QueryGenerator is an interface that is used
// to define queries that will be ran on the DB.
type QueryGenerator interface {
	QueryGenerate(f func() time.Time) (<-chan Query, error)
	SetTime(t time.Time)
}

// QueryClient is an interface that can write a query
// to an InfluxDB instance.
type QueryClient interface {
	Query(q Query) (response, error)
	Exec(qs <-chan Query, r chan<- response) error
}

// Querier queries the database.
type Querier struct {
	QueryGenerator
	QueryClient
}

// NewQuerier returns a Querier.
func NewQuerier(q QueryGenerator, c QueryClient) Querier {
	r := Querier{
		QueryGenerator: q,
		QueryClient:    c,
	}

	return r
}

///////////////////////////////////
// Definition of the Provisioner //
///////////////////////////////////

// Provisioner is an interface that provisions an
// InfluxDB instance
type Provisioner interface {
	Provision() error
}

/////////////////////////////////
// Definition of StressTest /////
/////////////////////////////////

// StressTest is a struct that contains all of
// the logic required to execute a Stress Test
type StressTest struct {
	Provisioner
	Writer
	Querier
}

// responseHandler
type responseHandler func(r <-chan response, t *Timer)

// Start executes the Stress Test
func (s *StressTest) Start(wHandle responseHandler, rHandle responseHandler) {
	var wg sync.WaitGroup

	// Provision the Instance
	s.Provision()

	wg.Add(1)
	// Starts Writing
	go func() {
		r := make(chan response, 0)
		wt := NewTimer()

		go func() {
			defer wt.StopTimer()
			defer close(r)
			p, err := s.Generate()
			if err != nil {
				fmt.Println(err)
				return
			}

			err = s.Batch(p, r)
			if err != nil {
				fmt.Println(err)
				return
			}
		}()

		// Write Results Handler
		wHandle(r, wt)
		wg.Done()
	}()

	wg.Add(1)
	// Starts Querying
	go func() {
		r := make(chan response, 0)
		rt := NewTimer()

		go func() {
			defer rt.StopTimer()
			defer close(r)
			q, err := s.QueryGenerate(s.Time)
			if err != nil {
				fmt.Println(err)
				return
			}

			err = s.Exec(q, r)
			if err != nil {
				fmt.Println(err)
				return
			}
		}()

		// Read Results Handler
		rHandle(r, rt)
		wg.Done()
	}()

	wg.Wait()
}

// NewStressTest returns an instance of a StressTest
func NewStressTest(p Provisioner, w Writer, r Querier) StressTest {
	s := StressTest{
		Provisioner: p,
		Writer:      w,
		Querier:     r,
	}

	return s
}
