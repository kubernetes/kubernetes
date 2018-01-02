package stressClient

import (
	"testing"
	"time"
)

func TestNewStressClientTags(t *testing.T) {
	pe, _, _ := newTestStressClient("localhost:8086")
	tags := pe.tags("foo_id")
	expected := fmtInt(len(pe.addresses))
	got := tags["number_targets"]
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
	expected = pe.precision
	got = tags["precision"]
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
	expected = pe.wdelay
	got = tags["write_interval"]
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
	expected = "foo_id"
	got = tags["statement_id"]
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
}

func TestNewStressTestTags(t *testing.T) {
	sf, _, _ := NewTestStressTest()
	tags := sf.tags()
	expected := sf.Precision
	got := tags["precision"]
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
	expected = fmtInt(sf.BatchSize)
	got = tags["batch_size"]
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
}

func TestWritePoint(t *testing.T) {
	pe, _, _ := newTestStressClient("localhost:8086")
	statementID := "foo_id"
	responseCode := 200
	responseTime := time.Duration(10 * time.Millisecond)
	addedTags := map[string]string{"foo_tag": "foo_tag_value"}
	writeBytes := 28051
	pt := pe.writePoint(1, statementID, responseCode, responseTime, addedTags, writeBytes)
	got := pt.Tags()["statement_id"]
	if statementID != got {
		t.Errorf("expected: %v\ngot: %v\n", statementID, got)
	}
	got2 := int(pt.Fields()["status_code"].(int64))
	if responseCode != got2 {
		t.Errorf("expected: %v\ngot: %v\n", responseCode, got2)
	}
	expected := "write"
	got = pt.Name()
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
}

func TestQueryPoint(t *testing.T) {
	pe, _, _ := newTestStressClient("localhost:8086")
	statementID := "foo_id"
	responseCode := 200
	body := []byte{12}
	responseTime := time.Duration(10 * time.Millisecond)
	addedTags := map[string]string{"foo_tag": "foo_tag_value"}
	pt := pe.queryPoint(statementID, body, responseCode, responseTime, addedTags)
	got := pt.Tags()["statement_id"]
	if statementID != got {
		t.Errorf("expected: %v\ngot: %v\n", statementID, got)
	}
	got2 := int(pt.Fields()["status_code"].(int64))
	if responseCode != got2 {
		t.Errorf("expected: %v\ngot: %v\n", responseCode, got2)
	}
	expected := "query"
	got = pt.Name()
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
}
