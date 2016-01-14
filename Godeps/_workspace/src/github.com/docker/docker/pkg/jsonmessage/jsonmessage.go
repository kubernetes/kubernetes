package jsonmessage

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/docker/docker/pkg/term"
	"github.com/docker/docker/pkg/timeutils"
	"github.com/docker/docker/pkg/units"
)

// JSONError wraps a concrete Code and Message, `Code` is
// is a integer error code, `Message` is the error message.
type JSONError struct {
	Code    int    `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
}

func (e *JSONError) Error() string {
	return e.Message
}

// JSONProgress describes a Progress. terminalFd is the fd of the current terminal,
// Start is the initial value for the operation. Current is the current status and
// value of the progress made towards Total. Total is the end value describing when
// we made 100% progress for an operation.
type JSONProgress struct {
	terminalFd uintptr
	Current    int64 `json:"current,omitempty"`
	Total      int64 `json:"total,omitempty"`
	Start      int64 `json:"start,omitempty"`
}

func (p *JSONProgress) String() string {
	var (
		width       = 200
		pbBox       string
		numbersBox  string
		timeLeftBox string
	)

	ws, err := term.GetWinsize(p.terminalFd)
	if err == nil {
		width = int(ws.Width)
	}

	if p.Current <= 0 && p.Total <= 0 {
		return ""
	}
	current := units.HumanSize(float64(p.Current))
	if p.Total <= 0 {
		return fmt.Sprintf("%8v", current)
	}
	total := units.HumanSize(float64(p.Total))
	percentage := int(float64(p.Current)/float64(p.Total)*100) / 2
	if percentage > 50 {
		percentage = 50
	}
	if width > 110 {
		// this number can't be negetive gh#7136
		numSpaces := 0
		if 50-percentage > 0 {
			numSpaces = 50 - percentage
		}
		pbBox = fmt.Sprintf("[%s>%s] ", strings.Repeat("=", percentage), strings.Repeat(" ", numSpaces))
	}

	numbersBox = fmt.Sprintf("%8v/%v", current, total)

	if p.Current > p.Total {
		// remove total display if the reported current is wonky.
		numbersBox = fmt.Sprintf("%8v", current)
	}

	if p.Current > 0 && p.Start > 0 && percentage < 50 {
		fromStart := time.Now().UTC().Sub(time.Unix(p.Start, 0))
		perEntry := fromStart / time.Duration(p.Current)
		left := time.Duration(p.Total-p.Current) * perEntry
		left = (left / time.Second) * time.Second

		if width > 50 {
			timeLeftBox = " " + left.String()
		}
	}
	return pbBox + numbersBox + timeLeftBox
}

// JSONMessage defines a message struct. It describes
// the created time, where it from, status, ID of the
// message. It's used for docker events.
type JSONMessage struct {
	Stream          string        `json:"stream,omitempty"`
	Status          string        `json:"status,omitempty"`
	Progress        *JSONProgress `json:"progressDetail,omitempty"`
	ProgressMessage string        `json:"progress,omitempty"` //deprecated
	ID              string        `json:"id,omitempty"`
	From            string        `json:"from,omitempty"`
	Time            int64         `json:"time,omitempty"`
	TimeNano        int64         `json:"timeNano,omitempty"`
	Error           *JSONError    `json:"errorDetail,omitempty"`
	ErrorMessage    string        `json:"error,omitempty"` //deprecated
}

// Display displays the JSONMessage to `out`. `isTerminal` describes if `out`
// is a terminal. If this is the case, it will erase the entire current line
// when dislaying the progressbar.
func (jm *JSONMessage) Display(out io.Writer, isTerminal bool) error {
	if jm.Error != nil {
		if jm.Error.Code == 401 {
			return fmt.Errorf("Authentication is required.")
		}
		return jm.Error
	}
	var endl string
	if isTerminal && jm.Stream == "" && jm.Progress != nil {
		// <ESC>[2K = erase entire current line
		fmt.Fprintf(out, "%c[2K\r", 27)
		endl = "\r"
	} else if jm.Progress != nil && jm.Progress.String() != "" { //disable progressbar in non-terminal
		return nil
	}
	if jm.TimeNano != 0 {
		fmt.Fprintf(out, "%s ", time.Unix(0, jm.TimeNano).Format(timeutils.RFC3339NanoFixed))
	} else if jm.Time != 0 {
		fmt.Fprintf(out, "%s ", time.Unix(jm.Time, 0).Format(timeutils.RFC3339NanoFixed))
	}
	if jm.ID != "" {
		fmt.Fprintf(out, "%s: ", jm.ID)
	}
	if jm.From != "" {
		fmt.Fprintf(out, "(from %s) ", jm.From)
	}
	if jm.Progress != nil && isTerminal {
		fmt.Fprintf(out, "%s %s%s", jm.Status, jm.Progress.String(), endl)
	} else if jm.ProgressMessage != "" { //deprecated
		fmt.Fprintf(out, "%s %s%s", jm.Status, jm.ProgressMessage, endl)
	} else if jm.Stream != "" {
		fmt.Fprintf(out, "%s%s", jm.Stream, endl)
	} else {
		fmt.Fprintf(out, "%s%s\n", jm.Status, endl)
	}
	return nil
}

// DisplayJSONMessagesStream displays a json message stream from `in` to `out`, `isTerminal`
// describes if `out` is a terminal. If this is the case, it will print `\n` at the end of
// each line and move the cursor while displaying.
func DisplayJSONMessagesStream(in io.Reader, out io.Writer, terminalFd uintptr, isTerminal bool) error {
	var (
		dec  = json.NewDecoder(in)
		ids  = make(map[string]int)
		diff = 0
	)
	for {
		var jm JSONMessage
		if err := dec.Decode(&jm); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		if jm.Progress != nil {
			jm.Progress.terminalFd = terminalFd
		}
		if jm.ID != "" && (jm.Progress != nil || jm.ProgressMessage != "") {
			line, ok := ids[jm.ID]
			if !ok {
				line = len(ids)
				ids[jm.ID] = line
				if isTerminal {
					fmt.Fprintf(out, "\n")
				}
				diff = 0
			} else {
				diff = len(ids) - line
			}
			if jm.ID != "" && isTerminal {
				// <ESC>[{diff}A = move cursor up diff rows
				fmt.Fprintf(out, "%c[%dA", 27, diff)
			}
		}
		err := jm.Display(out, isTerminal)
		if jm.ID != "" && isTerminal {
			// <ESC>[{diff}B = move cursor down diff rows
			fmt.Fprintf(out, "%c[%dB", 27, diff)
		}
		if err != nil {
			return err
		}
	}
	return nil
}
