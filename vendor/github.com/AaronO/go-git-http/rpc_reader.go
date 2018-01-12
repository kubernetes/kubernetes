package githttp

import (
	"io"
	"regexp"
	"strings"
)

// RpcReader scans for events in the incoming rpc request data.
type RpcReader struct {
	// Underlying reader (to relay calls to).
	io.Reader

	// Rpc type (receive-pack or upload-pack).
	Rpc string

	// List of events RpcReader has picked up through scanning.
	// These events do not have the Dir field set.
	Events []Event

	pktLineParser pktLineParser
}

// Read implements the io.Reader interface.
func (r *RpcReader) Read(p []byte) (n int, err error) {
	// Relay call
	n, err = r.Reader.Read(p)

	// Scan for events
	if n > 0 {
		r.scan(p[:n])
	}

	return n, err
}

func (r *RpcReader) scan(data []byte) {
	if r.pktLineParser.state == done {
		return
	}

	r.pktLineParser.Feed(data)

	// If parsing has just finished, process its output once.
	if r.pktLineParser.state == done {
		if r.pktLineParser.Error != nil {
			return
		}

		// When we get here, we're done collecting all pkt-lines successfully
		// and can now extract relevant events.
		switch r.Rpc {
		case "receive-pack":
			for _, line := range r.pktLineParser.Lines {
				events := scanPush(line)
				r.Events = append(r.Events, events...)
			}
		case "upload-pack":
			total := strings.Join(r.pktLineParser.Lines, "")
			events := scanFetch(total)
			r.Events = append(r.Events, events...)
		}
	}
}

// TODO: Avoid using regexp to parse a well documented binary protocol with an open source
//       implementation. There should not be a need for regexp.

// receivePackRegex is used once per pkt-line.
var receivePackRegex = regexp.MustCompile("([0-9a-fA-F]{40}) ([0-9a-fA-F]{40}) refs\\/(heads|tags)\\/(.+?)(\x00|$)")

func scanPush(line string) []Event {
	matches := receivePackRegex.FindAllStringSubmatch(line, -1)

	if matches == nil {
		return nil
	}

	var events []Event
	for _, m := range matches {
		e := Event{
			Last:   m[1],
			Commit: m[2],
		}

		// Handle pushes to branches and tags differently
		if m[3] == "heads" {
			e.Type = PUSH
			e.Branch = m[4]
		} else {
			e.Type = TAG
			e.Tag = m[4]
		}

		events = append(events, e)
	}

	return events
}

// uploadPackRegex is used once on the entire header data.
var uploadPackRegex = regexp.MustCompile(`^want ([0-9a-fA-F]{40})`)

func scanFetch(total string) []Event {
	matches := uploadPackRegex.FindAllStringSubmatch(total, -1)

	if matches == nil {
		return nil
	}

	var events []Event
	for _, m := range matches {
		events = append(events, Event{
			Type:   FETCH,
			Commit: m[1],
		})
	}

	return events
}
