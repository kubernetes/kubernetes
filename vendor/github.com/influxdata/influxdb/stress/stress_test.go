package stress

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/influxdata/influxdb/client/v2"
	"github.com/influxdata/influxdb/models"
)

func TestTimer_StartTimer(t *testing.T) {
	var epoch time.Time
	tmr := &Timer{}
	tmr.StartTimer()
	s := tmr.Start()
	if s == epoch {
		t.Errorf("expected tmr.start to not be %v", s)
	}
}

func TestNewTimer(t *testing.T) {
	var epoch time.Time
	tmr := NewTimer()
	s := tmr.Start()
	if s == epoch {
		t.Errorf("expected tmr.start to not be %v", s)
	}
	e := tmr.End()
	if e != epoch {
		t.Errorf("expected tmr.stop to be %v, got %v", epoch, e)
	}
}

func TestTimer_StopTimer(t *testing.T) {
	var epoch time.Time
	tmr := NewTimer()
	tmr.StopTimer()
	e := tmr.End()
	if e == epoch {
		t.Errorf("expected tmr.stop to not be %v", e)
	}
}

func TestTimer_Elapsed(t *testing.T) {
	tmr := NewTimer()
	time.Sleep(2 * time.Second)
	tmr.StopTimer()
	e := tmr.Elapsed()

	if time.Duration(1990*time.Millisecond) > e || e > time.Duration(3*time.Second) {
		t.Errorf("expected around %s got %s", time.Duration(2*time.Second), e)
	}
}

/// basic.go

// Types are off
func Test_typeArr(t *testing.T) {
	var re *regexp.Regexp
	var b bool
	arr := []string{
		"float64",
		"int",
		"bool",
	}

	ts := typeArr(arr)

	re = regexp.MustCompile(`[1-9]\d*`)
	b = re.MatchString(ts[0].(string))
	if !b {
		t.Errorf("Expected line protocol float64 got %v", ts[0])
	}

	re = regexp.MustCompile(`[1-9]\d*i`)
	b = re.MatchString(ts[1].(string))
	if !b {
		t.Errorf("Expected line protocol int got %v", ts[1])
	}

	re = regexp.MustCompile(`true|false`)
	b = re.MatchString(ts[2].(string))
	if !b {
		t.Errorf("Expected line protocol bool got %v", ts[2])
	}

}

func Test_typeArrBadTypes(t *testing.T) {
	arr := []string{
		"default",
		"rand",
		"",
	}

	ts := typeArr(arr)

	for _, x := range ts {
		re := regexp.MustCompile(`[1-9]\d*`)
		b := re.MatchString(x.(string))
		if !b {
			t.Errorf("Expected line protocol float64 got %v", x)
		}
	}
}

func TestPnt_Line(t *testing.T) {
	p := &Pnt{}
	b := []byte("a,b=1,c=1 v=1")

	p.Set(b)

	if string(p.Line()) != string(b) {
		t.Errorf("Expected `%v` to `%v`", string(b), string(p.Line()))
	}
}

func TestAbstractTags_Template(t *testing.T) {
	tags := AbstractTags{
		AbstractTag{
			Key:   "host",
			Value: "server",
		},
		AbstractTag{
			Key:   "location",
			Value: "us-west",
		},
	}

	s := tags.Template()
	tm := "host=server-%v,location=us-west"

	if s != tm {
		t.Errorf("Expected %v got %v", tm, s)
	}
}

func TestAbstractFields_TemplateOneField(t *testing.T) {
	fields := AbstractFields{
		AbstractField{
			Key:  "fValue",
			Type: "float64",
		},
	}

	tm, _ := fields.Template()

	s := "fValue=%v"
	if s != tm {
		t.Errorf("Expected `%v` got `%v`", s, tm)
	}

}

func TestAbstractFields_TemplateManyFields(t *testing.T) {
	fields := AbstractFields{
		AbstractField{
			Key:  "fValue",
			Type: "float64",
		},
		AbstractField{
			Key:  "iValue",
			Type: "int",
		},
		AbstractField{
			Key:  "bValue",
			Type: "bool",
		},
		AbstractField{
			Key:  "rValue",
			Type: "rnd",
		},
	}

	tm, ty := fields.Template()

	s := "fValue=%v,iValue=%v,bValue=%v,rValue=%v"
	if s != tm {
		t.Errorf("Expected `%v` got `%v`", s, tm)
	}

	for i, f := range fields {
		if f.Type != ty[i] {
			t.Errorf("Expected %v got %v", f.Type, ty[i])
		}
	}

}

var basicPG = &BasicPointGenerator{
	PointCount:  100,
	Tick:        "10s",
	Measurement: "cpu",
	SeriesCount: 100,
	Tags: AbstractTags{
		AbstractTag{
			Key:   "host",
			Value: "server",
		},
		AbstractTag{
			Key:   "location",
			Value: "us-west",
		},
	},
	Fields: AbstractFields{
		AbstractField{
			Key:  "value",
			Type: "float64",
		},
	},
	StartDate: "2006-Jan-01",
}

func TestBasicPointGenerator_Template(t *testing.T) {
	fn := basicPG.Template()
	now := time.Now()
	m := "cpu,host=server-1,location=us-west"
	ts := fmt.Sprintf("%v", now.UnixNano())

	tm := strings.Split(string(fn(1, now).Line()), " ")

	if m != tm[0] {
		t.Errorf("Expected %s got %s", m, tm[0])
	}

	if !strings.HasPrefix(tm[1], "value=") {
		t.Errorf("Expected %v to start with `value=`", tm[1])
	}

	if ts != string(tm[2]) {
		t.Errorf("Expected %s got %s", ts, tm[2])
	}
}

func TestBasicPointGenerator_Generate(t *testing.T) {
	ps, err := basicPG.Generate()
	if err != nil {
		t.Error(err)
	}

	var buf bytes.Buffer

	for p := range ps {
		b := p.Line()

		buf.Write(b)
		buf.Write([]byte("\n"))
	}

	bs := buf.Bytes()
	bs = bs[0 : len(bs)-1]

	_, err = models.ParsePoints(bs)
	if err != nil {
		t.Error(err)
	}
}

func Test_post(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		content, _ := ioutil.ReadAll(r.Body)
		lines := strings.Split(string(content), "\n")
		if len(lines) != 3 {
			t.Errorf("Expected 3 lines got %v", len(lines))
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	b := []byte(
		`cpu,host=server-1,location=us-west value=100 12932
		cpu,host=server-2,location=us-west value=10 12932
		cpu,host=server-3,location=us-west value=120 12932`,
	)

	_, err := post(ts.URL, "application/x-www-form-urlencoded", bytes.NewBuffer(b))
	if err != nil {
		t.Error(err)
	}
}

var basicIC = &BasicClient{
	Addresses:     []string{"localhost:8086"},
	Database:      "stress",
	Precision:     "n",
	BatchSize:     1000,
	BatchInterval: "0s",
	Concurrency:   10,
	Format:        "line_http",
}

func TestBasicClient_send(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		content, _ := ioutil.ReadAll(r.Body)
		lines := strings.Split(string(content), "\n")
		if len(lines) != 3 {
			t.Errorf("Expected 3 lines got %v", len(lines))
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	basicIC.Addresses[0] = ts.URL
	b := []byte(
		`cpu,host=server-1,location=us-west value=100 12932
		cpu,host=server-2,location=us-west value=10 12932
		cpu,host=server-3,location=us-west value=120 12932`,
	)
	_, err := basicIC.send(b)
	if err != nil {
		t.Error(err)
	}

}

func TestBasicClient_Batch(t *testing.T) {
	c := make(chan Point, 0)
	r := make(chan response, 0)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		content, _ := ioutil.ReadAll(r.Body)
		lines := strings.Split(string(content), "\n")
		if len(lines) != 1000 {
			t.Errorf("Expected 1000 lines got %v", len(lines))
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	basicIC.Addresses[0] = ts.URL[7:]

	go func(c chan Point) {
		defer close(c)

		for i := 0; i < 1000; i++ {
			p := &Pnt{}
			p.Next(i, time.Now())
			c <- *p
		}

	}(c)

	go func(r chan response) {
		for _ = range r {
		}
	}(r)

	err := basicIC.Batch(c, r)
	close(r)
	if err != nil {
		t.Error(err)
	}

}

var basicQ = &BasicQuery{
	Template:   Query("SELECT count(value) from cpu WHERE host='server-%v'"),
	QueryCount: 100,
}

func TestBasicQuery_QueryGenerate(t *testing.T) {
	qs, _ := basicQ.QueryGenerate(time.Now)

	i := 0
	for q := range qs {
		tm := fmt.Sprintf(string(basicQ.Template), i)
		if Query(tm) != q {
			t.Errorf("Expected %v to be %v", q, tm)
		}
		i++
	}
}

var basicQC = &BasicQueryClient{
	Addresses:     []string{"localhost:8086"},
	Database:      "stress",
	QueryInterval: "10s",
	Concurrency:   1,
}

func TestBasicQueryClient_Query(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(50 * time.Millisecond)
		w.Header().Set("X-Influxdb-Version", "x.x")
		var data client.Response
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(data)

		return
	}))
	defer ts.Close()

	basicQC.Addresses[0] = ts.URL[7:]
	basicQC.Init()

	q := "SELECT count(value) FROM cpu"
	r, err := basicQC.Query(Query(q))
	if err != nil {
		t.Error(err)
	}

	var epoch time.Time

	if r.Time == epoch {
		t.Errorf("Expected %v to not be epoch", r.Time)
	}

	elapsed := r.Timer.Elapsed()
	if elapsed.Nanoseconds() == 0 {
		t.Errorf("Expected %v to not be 0", elapsed.Nanoseconds())
	}

}

/// config.go
func Test_NewConfigWithFile(t *testing.T) {
	c, err := NewConfig("stress.toml")
	if err != nil {
		t.Error(err)
	}
	p := c.Provision
	w := c.Write
	r := c.Read

	if p.Basic.Address != "localhost:8086" {
		t.Errorf("Expected `localhost:8086` got %s", p.Basic.Address)
	}
	if p.Basic.Database != "stress" {
		t.Errorf("Expected `stress` got %s", p.Basic.Database)
	}
	if p.Basic.ResetDatabase != true {
		t.Errorf("Expected true got %v", p.Basic.ResetDatabase)
	}

	pg := w.PointGenerators.Basic
	if pg.PointCount != 100 {
		t.Errorf("Expected 100 got %v", pg.PointCount)
	}
	if pg.SeriesCount != 100000 {
		t.Errorf("Expected 100000 got %v", pg.SeriesCount)
	}
	if pg.Tick != "10s" {
		t.Errorf("Expected 10s got %s", pg.Tick)
	}
	if pg.Measurement != "cpu" {
		t.Errorf("Expected cpu got %s", pg.Measurement)
	}
	if pg.StartDate != "2006-Jan-02" {
		t.Errorf("Expected `2006-Jan-02` got `%s`", pg.StartDate)
	}
	// TODO: Check tags
	// TODO: Check fields

	wc := w.InfluxClients.Basic
	if wc.Addresses[0] != "localhost:8086" {
		t.Errorf("Expected `localhost:8086` got %s", wc.Addresses[0])
	}
	if wc.Database != "stress" {
		t.Errorf("Expected stress got %s", wc.Database)
	}
	if wc.Precision != "n" {
		t.Errorf("Expected n got %s", wc.Precision)
	}
	if wc.BatchSize != 10000 {
		t.Errorf("Expected 10000 got %v", wc.BatchSize)
	}
	if wc.BatchInterval != "0s" {
		t.Errorf("Expected 0s got %v", wc.BatchInterval)
	}
	if wc.Concurrency != 10 {
		t.Errorf("Expected 10 got %v", wc.Concurrency)
	}
	if wc.SSL != false {
		t.Errorf("Expected 10 got %v", wc.SSL)
	}
	if wc.Format != "line_http" {
		t.Errorf("Expected `line_http` got %s", wc.Format)
	}

	qg := r.QueryGenerators.Basic
	if qg.Template != "SELECT count(value) FROM cpu where host='server-%v'" {
		t.Errorf("Expected `SELECT count(value) FROM cpu where host='server-%%v'` got %s", qg.Template)
	}
	if qg.QueryCount != 250 {
		t.Errorf("Expected 250 got %v", qg.QueryCount)
	}

	qc := r.QueryClients.Basic
	if qc.Addresses[0] != "localhost:8086" {
		t.Errorf("Expected `localhost:8086` got %s", qc.Addresses[0])
	}
	if qc.Database != "stress" {
		t.Errorf("Expected stress got %s", qc.Database)
	}
	if qc.QueryInterval != "100ms" {
		t.Errorf("Expected 100ms got %s", qc.QueryInterval)
	}
	if qc.Concurrency != 1 {
		t.Errorf("Expected 1 got %v", qc.Concurrency)
	}
}

func Test_NewConfigWithoutFile(t *testing.T) {
	c, err := NewConfig("")
	if err != nil {
		t.Error(err)
	}
	p := c.Provision
	w := c.Write
	r := c.Read

	if p.Basic.Address != "localhost:8086" {
		t.Errorf("Expected `localhost:8086` got %s", p.Basic.Address)
	}
	if p.Basic.Database != "stress" {
		t.Errorf("Expected `stress` got %s", p.Basic.Database)
	}
	if p.Basic.ResetDatabase != true {
		t.Errorf("Expected true got %v", p.Basic.ResetDatabase)
	}

	pg := w.PointGenerators.Basic
	if pg.PointCount != 100 {
		t.Errorf("Expected 100 got %v", pg.PointCount)
	}
	if pg.SeriesCount != 100000 {
		t.Errorf("Expected 100000 got %v", pg.SeriesCount)
	}
	if pg.Tick != "10s" {
		t.Errorf("Expected 10s got %s", pg.Tick)
	}
	if pg.Measurement != "cpu" {
		t.Errorf("Expected cpu got %s", pg.Measurement)
	}
	if pg.StartDate != "2006-Jan-02" {
		t.Errorf("Expected `2006-Jan-02` got `%s`", pg.StartDate)
	}
	// TODO: Check tags
	// TODO: Check fields

	wc := w.InfluxClients.Basic
	if wc.Addresses[0] != "localhost:8086" {
		t.Errorf("Expected `localhost:8086` got %s", wc.Addresses[0])
	}
	if wc.Database != "stress" {
		t.Errorf("Expected stress got %s", wc.Database)
	}
	if wc.Precision != "n" {
		t.Errorf("Expected n got %s", wc.Precision)
	}
	if wc.BatchSize != 5000 {
		t.Errorf("Expected 5000 got %v", wc.BatchSize)
	}
	if wc.BatchInterval != "0s" {
		t.Errorf("Expected 0s got %v", wc.BatchInterval)
	}
	if wc.Concurrency != 10 {
		t.Errorf("Expected 10 got %v", wc.Concurrency)
	}
	if wc.SSL != false {
		t.Errorf("Expected 10 got %v", wc.SSL)
	}
	if wc.Format != "line_http" {
		t.Errorf("Expected `line_http` got %s", wc.Format)
	}

	qg := r.QueryGenerators.Basic
	if qg.Template != "SELECT count(value) FROM cpu where host='server-%v'" {
		t.Errorf("Expected `SELECT count(value) FROM cpu where host='server-%%v'` got %s", qg.Template)
	}
	if qg.QueryCount != 250 {
		t.Errorf("Expected 250 got %v", qg.QueryCount)
	}

	qc := r.QueryClients.Basic
	if qc.Addresses[0] != "localhost:8086" {
		t.Errorf("Expected `localhost:8086` got %s", qc.Addresses[0])
	}
	if qc.Database != "stress" {
		t.Errorf("Expected stress got %s", qc.Database)
	}
	if qc.QueryInterval != "100ms" {
		t.Errorf("Expected 100ms got %s", qc.QueryInterval)
	}
	if qc.Concurrency != 1 {
		t.Errorf("Expected 1 got %v", qc.Concurrency)
	}
}

/// run.go
// TODO
