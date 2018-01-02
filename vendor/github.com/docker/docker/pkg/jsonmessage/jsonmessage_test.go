package jsonmessage

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/docker/docker/pkg/jsonlog"
	"github.com/docker/docker/pkg/term"
)

func TestError(t *testing.T) {
	je := JSONError{404, "Not found"}
	if je.Error() != "Not found" {
		t.Fatalf("Expected 'Not found' got '%s'", je.Error())
	}
}

func TestProgress(t *testing.T) {
	termsz, err := term.GetWinsize(0)
	if err != nil {
		// we can safely ignore the err here
		termsz = nil
	}
	jp := JSONProgress{}
	if jp.String() != "" {
		t.Fatalf("Expected empty string, got '%s'", jp.String())
	}

	expected := "      1B"
	jp2 := JSONProgress{Current: 1}
	if jp2.String() != expected {
		t.Fatalf("Expected %q, got %q", expected, jp2.String())
	}

	expectedStart := "[==========>                                        ]      20B/100B"
	if termsz != nil && termsz.Width <= 110 {
		expectedStart = "    20B/100B"
	}
	jp3 := JSONProgress{Current: 20, Total: 100, Start: time.Now().Unix()}
	// Just look at the start of the string
	// (the remaining time is really hard to test -_-)
	if jp3.String()[:len(expectedStart)] != expectedStart {
		t.Fatalf("Expected to start with %q, got %q", expectedStart, jp3.String())
	}

	expected = "[=========================>                         ]      50B/100B"
	if termsz != nil && termsz.Width <= 110 {
		expected = "    50B/100B"
	}
	jp4 := JSONProgress{Current: 50, Total: 100}
	if jp4.String() != expected {
		t.Fatalf("Expected %q, got %q", expected, jp4.String())
	}

	// this number can't be negative gh#7136
	expected = "[==================================================>]      50B"
	if termsz != nil && termsz.Width <= 110 {
		expected = "    50B"
	}
	jp5 := JSONProgress{Current: 50, Total: 40}
	if jp5.String() != expected {
		t.Fatalf("Expected %q, got %q", expected, jp5.String())
	}

	expected = "[=========================>                         ] 50/100 units"
	if termsz != nil && termsz.Width <= 110 {
		expected = "    50/100 units"
	}
	jp6 := JSONProgress{Current: 50, Total: 100, Units: "units"}
	if jp6.String() != expected {
		t.Fatalf("Expected %q, got %q", expected, jp6.String())
	}

	// this number can't be negative
	expected = "[==================================================>] 50 units"
	if termsz != nil && termsz.Width <= 110 {
		expected = "    50 units"
	}
	jp7 := JSONProgress{Current: 50, Total: 40, Units: "units"}
	if jp7.String() != expected {
		t.Fatalf("Expected %q, got %q", expected, jp7.String())
	}

	expected = "[=========================>                         ] "
	if termsz != nil && termsz.Width <= 110 {
		expected = ""
	}
	jp8 := JSONProgress{Current: 50, Total: 100, HideCounts: true}
	if jp8.String() != expected {
		t.Fatalf("Expected %q, got %q", expected, jp8.String())
	}
}

func TestJSONMessageDisplay(t *testing.T) {
	now := time.Now()
	messages := map[JSONMessage][]string{
		// Empty
		{}: {"\n", "\n"},
		// Status
		{
			Status: "status",
		}: {
			"status\n",
			"status\n",
		},
		// General
		{
			Time:   now.Unix(),
			ID:     "ID",
			From:   "From",
			Status: "status",
		}: {
			fmt.Sprintf("%v ID: (from From) status\n", time.Unix(now.Unix(), 0).Format(jsonlog.RFC3339NanoFixed)),
			fmt.Sprintf("%v ID: (from From) status\n", time.Unix(now.Unix(), 0).Format(jsonlog.RFC3339NanoFixed)),
		},
		// General, with nano precision time
		{
			TimeNano: now.UnixNano(),
			ID:       "ID",
			From:     "From",
			Status:   "status",
		}: {
			fmt.Sprintf("%v ID: (from From) status\n", time.Unix(0, now.UnixNano()).Format(jsonlog.RFC3339NanoFixed)),
			fmt.Sprintf("%v ID: (from From) status\n", time.Unix(0, now.UnixNano()).Format(jsonlog.RFC3339NanoFixed)),
		},
		// General, with both times Nano is preferred
		{
			Time:     now.Unix(),
			TimeNano: now.UnixNano(),
			ID:       "ID",
			From:     "From",
			Status:   "status",
		}: {
			fmt.Sprintf("%v ID: (from From) status\n", time.Unix(0, now.UnixNano()).Format(jsonlog.RFC3339NanoFixed)),
			fmt.Sprintf("%v ID: (from From) status\n", time.Unix(0, now.UnixNano()).Format(jsonlog.RFC3339NanoFixed)),
		},
		// Stream over status
		{
			Status: "status",
			Stream: "stream",
		}: {
			"stream",
			"stream",
		},
		// With progress message
		{
			Status:          "status",
			ProgressMessage: "progressMessage",
		}: {
			"status progressMessage",
			"status progressMessage",
		},
		// With progress, stream empty
		{
			Status:   "status",
			Stream:   "",
			Progress: &JSONProgress{Current: 1},
		}: {
			"",
			fmt.Sprintf("%c[1K%c[K\rstatus       1B\r", 27, 27),
		},
	}

	// The tests :)
	for jsonMessage, expectedMessages := range messages {
		// Without terminal
		data := bytes.NewBuffer([]byte{})
		if err := jsonMessage.Display(data, nil); err != nil {
			t.Fatal(err)
		}
		if data.String() != expectedMessages[0] {
			t.Fatalf("Expected %q,got %q", expectedMessages[0], data.String())
		}
		// With terminal
		data = bytes.NewBuffer([]byte{})
		if err := jsonMessage.Display(data, &noTermInfo{}); err != nil {
			t.Fatal(err)
		}
		if data.String() != expectedMessages[1] {
			t.Fatalf("\nExpected %q\n     got %q", expectedMessages[1], data.String())
		}
	}
}

// Test JSONMessage with an Error. It will return an error with the text as error, not the meaning of the HTTP code.
func TestJSONMessageDisplayWithJSONError(t *testing.T) {
	data := bytes.NewBuffer([]byte{})
	jsonMessage := JSONMessage{Error: &JSONError{404, "Can't find it"}}

	err := jsonMessage.Display(data, &noTermInfo{})
	if err == nil || err.Error() != "Can't find it" {
		t.Fatalf("Expected a JSONError 404, got %q", err)
	}

	jsonMessage = JSONMessage{Error: &JSONError{401, "Anything"}}
	err = jsonMessage.Display(data, &noTermInfo{})
	if err == nil || err.Error() != "Authentication is required." {
		t.Fatalf("Expected an error \"Authentication is required.\", got %q", err)
	}
}

func TestDisplayJSONMessagesStreamInvalidJSON(t *testing.T) {
	var (
		inFd uintptr
	)
	data := bytes.NewBuffer([]byte{})
	reader := strings.NewReader("This is not a 'valid' JSON []")
	inFd, _ = term.GetFdInfo(reader)

	if err := DisplayJSONMessagesStream(reader, data, inFd, false, nil); err == nil && err.Error()[:17] != "invalid character" {
		t.Fatalf("Should have thrown an error (invalid character in ..), got %q", err)
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
			fmt.Sprintf("ID: status\n"),
		},
		// With progress
		"{ \"id\": \"ID\", \"status\": \"status\", \"progress\": \"ProgressMessage\" }": {
			"ID: status ProgressMessage",
			fmt.Sprintf("\n%c[%dAID: status ProgressMessage%c[%dB", 27, 1, 27, 1),
		},
		// With progressDetail
		"{ \"id\": \"ID\", \"status\": \"status\", \"progressDetail\": { \"Current\": 1} }": {
			"", // progressbar is disabled in non-terminal
			fmt.Sprintf("\n%c[%dA%c[1K%c[K\rID: status       1B\r%c[%dB", 27, 1, 27, 27, 27, 1),
		},
	}

	// Use $TERM which is unlikely to exist, forcing DisplayJSONMessageStream to
	// (hopefully) use &noTermInfo.
	origTerm := os.Getenv("TERM")
	os.Setenv("TERM", "xyzzy-non-existent-terminfo")

	for jsonMessage, expectedMessages := range messages {
		data := bytes.NewBuffer([]byte{})
		reader := strings.NewReader(jsonMessage)
		inFd, _ = term.GetFdInfo(reader)

		// Without terminal
		if err := DisplayJSONMessagesStream(reader, data, inFd, false, nil); err != nil {
			t.Fatal(err)
		}
		if data.String() != expectedMessages[0] {
			t.Fatalf("Expected an %q, got %q", expectedMessages[0], data.String())
		}

		// With terminal
		data = bytes.NewBuffer([]byte{})
		reader = strings.NewReader(jsonMessage)
		if err := DisplayJSONMessagesStream(reader, data, inFd, true, nil); err != nil {
			t.Fatal(err)
		}
		if data.String() != expectedMessages[1] {
			t.Fatalf("\nExpected %q\n     got %q", expectedMessages[1], data.String())
		}
	}
	os.Setenv("TERM", origTerm)

}
