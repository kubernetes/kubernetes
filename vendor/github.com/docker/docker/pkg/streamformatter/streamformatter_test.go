package streamformatter

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/docker/docker/pkg/jsonmessage"
)

func TestFormatStream(t *testing.T) {
	sf := NewStreamFormatter()
	res := sf.FormatStream("stream")
	if string(res) != "stream"+"\r" {
		t.Fatalf("%q", res)
	}
}

func TestFormatJSONStatus(t *testing.T) {
	sf := NewStreamFormatter()
	res := sf.FormatStatus("ID", "%s%d", "a", 1)
	if string(res) != "a1\r\n" {
		t.Fatalf("%q", res)
	}
}

func TestFormatSimpleError(t *testing.T) {
	sf := NewStreamFormatter()
	res := sf.FormatError(errors.New("Error for formatter"))
	if string(res) != "Error: Error for formatter\r\n" {
		t.Fatalf("%q", res)
	}
}

func TestJSONFormatStream(t *testing.T) {
	sf := NewJSONStreamFormatter()
	res := sf.FormatStream("stream")
	if string(res) != `{"stream":"stream"}`+"\r\n" {
		t.Fatalf("%q", res)
	}
}

func TestJSONFormatStatus(t *testing.T) {
	sf := NewJSONStreamFormatter()
	res := sf.FormatStatus("ID", "%s%d", "a", 1)
	if string(res) != `{"status":"a1","id":"ID"}`+"\r\n" {
		t.Fatalf("%q", res)
	}
}

func TestJSONFormatSimpleError(t *testing.T) {
	sf := NewJSONStreamFormatter()
	res := sf.FormatError(errors.New("Error for formatter"))
	if string(res) != `{"errorDetail":{"message":"Error for formatter"},"error":"Error for formatter"}`+"\r\n" {
		t.Fatalf("%q", res)
	}
}

func TestJSONFormatJSONError(t *testing.T) {
	sf := NewJSONStreamFormatter()
	err := &jsonmessage.JSONError{Code: 50, Message: "Json error"}
	res := sf.FormatError(err)
	if string(res) != `{"errorDetail":{"code":50,"message":"Json error"},"error":"Json error"}`+"\r\n" {
		t.Fatalf("%q", res)
	}
}

func TestJSONFormatProgress(t *testing.T) {
	sf := NewJSONStreamFormatter()
	progress := &jsonmessage.JSONProgress{
		Current: 15,
		Total:   30,
		Start:   1,
	}
	res := sf.FormatProgress("id", "action", progress)
	msg := &jsonmessage.JSONMessage{}
	if err := json.Unmarshal(res, msg); err != nil {
		t.Fatal(err)
	}
	if msg.ID != "id" {
		t.Fatalf("ID must be 'id', got: %s", msg.ID)
	}
	if msg.Status != "action" {
		t.Fatalf("Status must be 'action', got: %s", msg.Status)
	}
	if msg.ProgressMessage != progress.String() {
		t.Fatalf("ProgressMessage must be %s, got: %s", progress.String(), msg.ProgressMessage)
	}
	if !reflect.DeepEqual(msg.Progress, progress) {
		t.Fatal("Original progress not equals progress from FormatProgress")
	}
}
