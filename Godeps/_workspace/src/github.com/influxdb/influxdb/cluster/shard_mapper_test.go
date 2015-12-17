package cluster

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"testing"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/tsdb"
)

// remoteShardResponder implements the remoteShardConn interface.
type remoteShardResponder struct {
	net.Conn
	t       *testing.T
	rxBytes []byte

	buffer *bytes.Buffer
}

func newRemoteShardResponder(outputs []*tsdb.MapperOutput, tagsets []string) *remoteShardResponder {
	r := &remoteShardResponder{}
	a := make([]byte, 0, 1024)
	r.buffer = bytes.NewBuffer(a)

	// Pump the outputs in the buffer for later reading.
	for _, o := range outputs {
		resp := &MapShardResponse{}
		resp.SetCode(0)
		if o != nil {
			d, _ := json.Marshal(o)
			resp.SetData(d)
			resp.SetTagSets(tagsets)
		}

		g, _ := resp.MarshalBinary()
		WriteTLV(r.buffer, mapShardResponseMessage, g)
	}

	return r
}

func (r remoteShardResponder) Close() error { return nil }
func (r remoteShardResponder) Read(p []byte) (n int, err error) {
	return io.ReadFull(r.buffer, p)
}

func (r remoteShardResponder) Write(p []byte) (n int, err error) {
	if r.rxBytes == nil {
		r.rxBytes = make([]byte, 0)
	}
	r.rxBytes = append(r.rxBytes, p...)
	return len(p), nil
}

// Ensure a RemoteMapper can process valid responses from a remote shard.
func TestShardWriter_RemoteMapper_Success(t *testing.T) {
	expTagSets := []string{"tagsetA"}
	expOutput := &tsdb.MapperOutput{
		Name: "cpu",
		Tags: map[string]string{"host": "serverA"},
	}

	c := newRemoteShardResponder([]*tsdb.MapperOutput{expOutput, nil}, expTagSets)

	r := NewRemoteMapper(c, 1234, mustParseStmt("SELECT * FROM CPU"), 10)
	if err := r.Open(); err != nil {
		t.Fatalf("failed to open remote mapper: %s", err.Error())
	}

	if r.TagSets()[0] != expTagSets[0] {
		t.Fatalf("incorrect tagsets received, exp %v, got %v", expTagSets, r.TagSets())
	}

	// Get first chunk from mapper.
	chunk, err := r.NextChunk()
	if err != nil {
		t.Fatalf("failed to get next chunk from mapper: %s", err.Error())
	}
	output, ok := chunk.(*tsdb.MapperOutput)
	if !ok {
		t.Fatal("chunk is not of expected type")
	}
	if output.Name != "cpu" {
		t.Fatalf("received output incorrect, exp: %v, got %v", expOutput, output)
	}

	// Next chunk should be nil, indicating no more data.
	chunk, err = r.NextChunk()
	if err != nil {
		t.Fatalf("failed to get next chunk from mapper: %s", err.Error())
	}
	if chunk != nil {
		t.Fatal("received more chunks when none expected")
	}
}

// mustParseStmt parses a single statement or panics.
func mustParseStmt(stmt string) influxql.Statement {
	q, err := influxql.ParseQuery(stmt)
	if err != nil {
		panic(err)
	} else if len(q.Statements) != 1 {
		panic(fmt.Sprintf("expected 1 statement but got %d", len(q.Statements)))
	}
	return q.Statements[0]
}
