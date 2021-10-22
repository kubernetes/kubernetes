// +build integration,perftest

package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"
)

type Logger struct {
	out *csv.Writer
}

func NewLogger(writer io.Writer) *Logger {
	l := &Logger{
		out: csv.NewWriter(writer),
	}

	err := l.out.Write([]string{
		"ID", "Attempt",
		"Latency",
		"DNSStart", "DNSDone", "DNSDur",
		"ConnectStart", "ConnectDone", "ConnectDur",
		"TLSStart", "TLSDone", "TLSDur",
		"WriteReq", "RespFirstByte", "WaitRespFirstByte",
		"Error",
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to write header trace, %v\n", err)
	}

	return l
}

func (l *Logger) RecordTrace(trace *RequestTrace) {
	req := RequestReport{
		ID:           trace.ID,
		TotalLatency: durToMSString(trace.TotalLatency()),
		Retries:      trace.Retries(),
	}

	for i, a := range trace.Attempts() {
		attempt := AttemptReport{
			Reused:     a.Reused,
			SDKMarshal: durToMSString(a.SendStart.Sub(a.Start)),
			ReqWritten: durToMSString(a.RequestWritten.Sub(a.SendStart)),
			Latency:    durToMSString(a.Finish.Sub(a.Start)),
			Err:        a.Err,
		}

		if !a.FirstResponseByte.IsZero() {
			attempt.RespFirstByte = durToMSString(a.FirstResponseByte.Sub(a.SendStart))
			attempt.WaitRespFirstByte = durToMSString(a.FirstResponseByte.Sub(a.RequestWritten))
		}

		if !a.Reused {
			attempt.DNSStart = durToMSString(a.DNSStart.Sub(a.SendStart))
			attempt.DNSDone = durToMSString(a.DNSDone.Sub(a.SendStart))
			attempt.DNS = durToMSString(a.DNSDone.Sub(a.DNSStart))

			attempt.ConnectStart = durToMSString(a.ConnectStart.Sub(a.SendStart))
			attempt.ConnectDone = durToMSString(a.ConnectDone.Sub(a.SendStart))
			attempt.Connect = durToMSString(a.ConnectDone.Sub(a.ConnectStart))

			attempt.TLSHandshakeStart = durToMSString(a.TLSHandshakeStart.Sub(a.SendStart))
			attempt.TLSHandshakeDone = durToMSString(a.TLSHandshakeDone.Sub(a.SendStart))
			attempt.TLSHandshake = durToMSString(a.TLSHandshakeDone.Sub(a.TLSHandshakeStart))
		}

		req.Attempts = append(req.Attempts, attempt)

		var reqErr string
		if attempt.Err != nil {
			reqErr = strings.Replace(attempt.Err.Error(), "\n", `\n`, -1)
		}
		err := l.out.Write([]string{
			strconv.Itoa(int(req.ID)),
			strconv.Itoa(i + 1),
			attempt.Latency,
			attempt.DNSStart, attempt.DNSDone, attempt.DNS,
			attempt.ConnectStart, attempt.ConnectDone, attempt.Connect,
			attempt.TLSHandshakeStart, attempt.TLSHandshakeDone, attempt.TLSHandshake,
			attempt.ReqWritten,
			attempt.RespFirstByte,
			attempt.WaitRespFirstByte,
			reqErr,
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to write request trace, %v\n", err)
		}
		l.out.Flush()
	}
}

func durToMSString(v time.Duration) string {
	ms := float64(v) / float64(time.Millisecond)
	return fmt.Sprintf("%0.6f", ms)
}

type RequestReport struct {
	ID           int64
	TotalLatency string
	Retries      int

	Attempts []AttemptReport
}
type AttemptReport struct {
	Reused bool
	Err    error

	SDKMarshal string `json:",omitempty"`

	DNSStart string `json:",omitempty"`
	DNSDone  string `json:",omitempty"`
	DNS      string `json:",omitempty"`

	ConnectStart string `json:",omitempty"`
	ConnectDone  string `json:",omitempty"`
	Connect      string `json:",omitempty"`

	TLSHandshakeStart string `json:",omitempty"`
	TLSHandshakeDone  string `json:",omitempty"`
	TLSHandshake      string `json:",omitempty"`

	ReqWritten        string `json:",omitempty"`
	RespFirstByte     string `json:",omitempty"`
	WaitRespFirstByte string `json:",omitempty"`
	Latency           string `json:",omitempty"`
}
