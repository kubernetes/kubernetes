package datadog

import (
	"fmt"
	"net"
	"reflect"
	"testing"
)

var EmptyTags []string

const (
	DogStatsdAddr    = "127.0.0.1:7254"
	HostnameEnabled  = true
	HostnameDisabled = false
	TestHostname     = "test_hostname"
)

func MockGetHostname() string {
	return TestHostname
}

var ParseKeyTests = []struct {
	KeyToParse        []string
	Tags              []string
	PropagateHostname bool
	ExpectedKey       []string
	ExpectedTags      []string
}{
	{[]string{"a", MockGetHostname(), "b", "c"}, EmptyTags, HostnameDisabled, []string{"a", "b", "c"}, EmptyTags},
	{[]string{"a", "b", "c"}, EmptyTags, HostnameDisabled, []string{"a", "b", "c"}, EmptyTags},
	{[]string{"a", "b", "c"}, EmptyTags, HostnameEnabled, []string{"a", "b", "c"}, []string{fmt.Sprintf("host:%s", MockGetHostname())}},
}

var FlattenKeyTests = []struct {
	KeyToFlatten []string
	Expected     string
}{
	{[]string{"a", "b", "c"}, "a.b.c"},
	{[]string{"spaces must", "flatten", "to", "underscores"}, "spaces_must.flatten.to.underscores"},
}

var MetricSinkTests = []struct {
	Method            string
	Metric            []string
	Value             interface{}
	Tags              []string
	PropagateHostname bool
	Expected          string
}{
	{"SetGauge", []string{"foo", "bar"}, float32(42), EmptyTags, HostnameDisabled, "foo.bar:42.000000|g"},
	{"SetGauge", []string{"foo", "bar", "baz"}, float32(42), EmptyTags, HostnameDisabled, "foo.bar.baz:42.000000|g"},
	{"AddSample", []string{"sample", "thing"}, float32(4), EmptyTags, HostnameDisabled, "sample.thing:4.000000|ms"},
	{"IncrCounter", []string{"count", "me"}, float32(3), EmptyTags, HostnameDisabled, "count.me:3|c"},

	{"SetGauge", []string{"foo", "baz"}, float32(42), []string{"my_tag:my_value"}, HostnameDisabled, "foo.baz:42.000000|g|#my_tag:my_value"},
	{"SetGauge", []string{"foo", "bar"}, float32(42), []string{"my_tag:my_value", "other_tag:other_value"}, HostnameDisabled, "foo.bar:42.000000|g|#my_tag:my_value,other_tag:other_value"},
	{"SetGauge", []string{"foo", "bar"}, float32(42), []string{"my_tag:my_value", "other_tag:other_value"}, HostnameEnabled, "foo.bar:42.000000|g|#my_tag:my_value,other_tag:other_value,host:test_hostname"},
}

func mockNewDogStatsdSink(addr string, tags []string, tagWithHostname bool) *DogStatsdSink {
	dog, _ := NewDogStatsdSink(addr, MockGetHostname())
	dog.SetTags(tags)
	if tagWithHostname {
		dog.EnableHostNamePropagation()
	}

	return dog
}

func setupTestServerAndBuffer(t *testing.T) (*net.UDPConn, []byte) {
	udpAddr, err := net.ResolveUDPAddr("udp", DogStatsdAddr)
	if err != nil {
		t.Fatal(err)
	}
	server, err := net.ListenUDP("udp", udpAddr)
	if err != nil {
		t.Fatal(err)
	}
	return server, make([]byte, 1024)
}

func TestParseKey(t *testing.T) {
	for _, tt := range ParseKeyTests {
		dog := mockNewDogStatsdSink(DogStatsdAddr, tt.Tags, tt.PropagateHostname)
		key, tags := dog.parseKey(tt.KeyToParse)

		if !reflect.DeepEqual(key, tt.ExpectedKey) {
			t.Fatalf("Key Parsing failed for %v", tt.KeyToParse)
		}

		if !reflect.DeepEqual(tags, tt.ExpectedTags) {
			t.Fatalf("Tag Parsing Failed for %v", tt.KeyToParse)
		}
	}
}

func TestFlattenKey(t *testing.T) {
	dog := mockNewDogStatsdSink(DogStatsdAddr, EmptyTags, HostnameDisabled)
	for _, tt := range FlattenKeyTests {
		if !reflect.DeepEqual(dog.flattenKey(tt.KeyToFlatten), tt.Expected) {
			t.Fatalf("Flattening %v failed", tt.KeyToFlatten)
		}
	}
}

func TestMetricSink(t *testing.T) {
	server, buf := setupTestServerAndBuffer(t)
	defer server.Close()

	for _, tt := range MetricSinkTests {
		dog := mockNewDogStatsdSink(DogStatsdAddr, tt.Tags, tt.PropagateHostname)
		method := reflect.ValueOf(dog).MethodByName(tt.Method)
		method.Call([]reflect.Value{
			reflect.ValueOf(tt.Metric),
			reflect.ValueOf(tt.Value)})
		assertServerMatchesExpected(t, server, buf, tt.Expected)
	}
}

func TestTaggableMetrics(t *testing.T) {
	server, buf := setupTestServerAndBuffer(t)
	defer server.Close()

	dog := mockNewDogStatsdSink(DogStatsdAddr, EmptyTags, HostnameDisabled)

	dog.AddSampleWithTags([]string{"sample", "thing"}, float32(4), []string{"tagkey:tagvalue"})
	assertServerMatchesExpected(t, server, buf, "sample.thing:4.000000|ms|#tagkey:tagvalue")

	dog.SetGaugeWithTags([]string{"sample", "thing"}, float32(4), []string{"tagkey:tagvalue"})
	assertServerMatchesExpected(t, server, buf, "sample.thing:4.000000|g|#tagkey:tagvalue")

	dog.IncrCounterWithTags([]string{"sample", "thing"}, float32(4), []string{"tagkey:tagvalue"})
	assertServerMatchesExpected(t, server, buf, "sample.thing:4|c|#tagkey:tagvalue")

	dog = mockNewDogStatsdSink(DogStatsdAddr, []string{"global"}, HostnameEnabled) // with hostname, global tags
	dog.IncrCounterWithTags([]string{"sample", "thing"}, float32(4), []string{"tagkey:tagvalue"})
	assertServerMatchesExpected(t, server, buf, "sample.thing:4|c|#global,tagkey:tagvalue,host:test_hostname")
}

func assertServerMatchesExpected(t *testing.T, server *net.UDPConn, buf []byte, expected string) {
	n, _ := server.Read(buf)
	msg := buf[:n]
	if string(msg) != expected {
		t.Fatalf("Line %s does not match expected: %s", string(msg), expected)
	}
}
