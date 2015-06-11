package jsonmessage

import (
	"bytes"
	"fmt"
	"testing"
	"time"

	"github.com/docker/docker/pkg/term"
	"github.com/docker/docker/pkg/timeutils"
	"strings"
)

func TestError(t *testing.T) {
	je := JSONError{404, "Not found"}
	if je.Error() != "Not found" {
		t.Fatalf("Expected 'Not found' got '%s'", je.Error())
	}
}

func TestProgress(t *testing.T) {
	jp := JSONProgress{}
	if jp.String() != "" {
		t.Fatalf("Expected empty string, got '%s'", jp.String())
	}

	expected := "     1 B"
	jp2 := JSONProgress{Current: 1}
	if jp2.String() != expected {
		t.Fatalf("Expected %q, got %q", expected, jp2.String())
	}

	expectedStart := "[==========>                                        ]     20 B/100 B"
	jp3 := JSONProgress{Current: 20, Total: 100, Start: time.Now().Unix()}
	// Just look at the start of the string
	// (the remaining time is really hard to test -_-)
	if jp3.String()[:len(expectedStart)] != expectedStart {
		t.Fatalf("Expected to start with %q, got %q", expectedStart, jp3.String())
	}

	expected = "[=========================>                         ]     50 B/100 B"
	jp4 := JSONProgress{Current: 50, Total: 100}
	if jp4.String() != expected {
		t.Fatalf("Expected %q, got %q", expected, jp4.String())
	}

	// this number can't be negative gh#7136
	expected = "[==================================================>]     50 B/40 B"
	jp5 := JSONProgress{Current: 50, Total: 40}
	if jp5.String() != expected {
		t.Fatalf("Expected %q, got %q", expected, jp5.String())
	}
}

func TestJSONMessageDisplay(t *testing.T) {
	now := time.Now().Unix()
	messages := map[JSONMessage][]string{
		// Empty
		JSONMessage{}: {"\n", "\n"},
		// Status
		JSONMessage{
			Status: "status",
		}: {
			"status\n",
			"status\n",
		},
		// General
		JSONMessage{
			Time:   now,
			ID:     "ID",
			From:   "From",
			Status: "status",
		}: {
			fmt.Sprintf("%v ID: (from From) status\n", time.Unix(now, 0).Format(timeutils.RFC3339NanoFixed)),
			fmt.Sprintf("%v ID: (from From) status\n", time.Unix(now, 0).Format(timeutils.RFC3339NanoFixed)),
		},
		// Stream over status
		JSONMessage{
			Status: "status",
			Stream: "stream",
		}: {
			"stream",
			"stream",
		},
		// With progress message
		JSONMessage{
			Status:          "status",
			ProgressMessage: "progressMessage",
		}: {
			"status progressMessage",
			"status progressMessage",
		},
		// With progress, stream empty
		JSONMessage{
			Status:   "status",
			Stream:   "",
			Progress: &JSONProgress{Current: 1},
		}: {
			"",
			fmt.Sprintf("%c[2K\rstatus      1 B\r", 27),
		},
	}

	// The tests :)
	for jsonMessage, expectedMessages := range messages {
		// Without terminal
		data := bytes.NewBuffer([]byte{})
		if err := jsonMessage.Display(data, false); err != nil {
			t.Fatal(err)
		}
		if data.String() != expectedMessages[0] {
			t.Fatalf("Expected [%v], got [%v]", expectedMessages[0], data.String())
		}
		// With terminal
		data = bytes.NewBuffer([]byte{})
		if err := jsonMessage.Display(data, true); err != nil {
			t.Fatal(err)
		}
		if data.String() != expectedMessages[1] {
			t.Fatalf("Expected [%v], got [%v]", expectedMessages[1], data.String())
		}
	}
}

// Test JSONMessage with an Error. It will return an error with the text as error, not the meaning of the HTTP code.
func TestJSONMessageDisplayWithJSONError(t *testing.T) {
	data := bytes.NewBuffer([]byte{})
	jsonMessage := JSONMessage{Error: &JSONError{404, "Can't find it"}}

	err := jsonMessage.Display(data, true)
	if err == nil || err.Error() != "Can't find it" {
		t.Fatalf("Expected a JSONError 404, got [%v]", err)
	}

	jsonMessage = JSONMessage{Error: &JSONError{401, "Anything"}}
	err = jsonMessage.Display(data, true)
	if err == nil || err.Error() != "Authentication is required." {
		t.Fatalf("Expected an error [Authentication is required.], got [%v]", err)
	}
}

func TestDisplayJSONMessagesStreamInvalidJSON(t *testing.T) {
	var (
		inFd uintptr
	)
	data := bytes.NewBuffer([]byte{})
	reader := strings.NewReader("This is not a 'valid' JSON []")
	inFd, _ = term.GetFdInfo(reader)

	if err := DisplayJSONMessagesStream(reader, data, inFd, false); err == nil && err.Error()[:17] != "invalid character" {
		t.Fatalf("Should have thrown an error (invalid character in ..), got [%v]", err)
	}
}

func TestDisplayJSONMessagesStream(t *testing.T) {
	var (
		inFd uintptr
	)

	messages := map[string][]string{
		// empty string
		"": {
			"",
			""},
		// Without progress & ID
		"{ \"status\": \"status\" }": {
			"status\n",
			"status\n",
		},
		// Without progress, with ID
		"{ \"id\": \"ID\",\"status\": \"status\" }": {
			"ID: status\n",
			fmt.Sprintf("ID: status\n%c[%dB", 27, 0),
		},
		// With progress
		"{ \"id\": \"ID\", \"status\": \"status\", \"progress\": \"ProgressMessage\" }": {
			"ID: status ProgressMessage",
			fmt.Sprintf("\n%c[%dAID: status ProgressMessage%c[%dB", 27, 0, 27, 0),
		},
		// With progressDetail
		"{ \"id\": \"ID\", \"status\": \"status\", \"progressDetail\": { \"Current\": 1} }": {
			"", // progressbar is disabled in non-terminal
			fmt.Sprintf("\n%c[%dA%c[2K\rID: status      1 B\r%c[%dB", 27, 0, 27, 27, 0),
		},
	}
	for jsonMessage, expectedMessages := range messages {
		data := bytes.NewBuffer([]byte{})
		reader := strings.NewReader(jsonMessage)
		inFd, _ = term.GetFdInfo(reader)

		// Without terminal
		if err := DisplayJSONMessagesStream(reader, data, inFd, false); err != nil {
			t.Fatal(err)
		}
		if data.String() != expectedMessages[0] {
			t.Fatalf("Expected an [%v], got [%v]", expectedMessages[0], data.String())
		}

		// With terminal
		data = bytes.NewBuffer([]byte{})
		reader = strings.NewReader(jsonMessage)
		if err := DisplayJSONMessagesStream(reader, data, inFd, true); err != nil {
			t.Fatal(err)
		}
		if data.String() != expectedMessages[1] {
			t.Fatalf("Expected an [%v], got [%v]", expectedMessages[1], data.String())
		}
	}

}
