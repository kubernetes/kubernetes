package jsonlog

import (
	"encoding/json"
	"fmt"
	"time"
)

// JSONLog represents a log message, typically a single entry from a given log stream.
// JSONLogs can be easily serialized to and from JSON and support custom formatting.
type JSONLog struct {
	// Log is the log message
	Log string `json:"log,omitempty"`
	// Stream is the log source
	Stream string `json:"stream,omitempty"`
	// Created is the created timestamp of log
	Created time.Time `json:"time"`
}

// Format returns the log formatted according to format
// If format is nil, returns the log message
// If format is json, returns the log marshaled in json format
// By default, returns the log with the log time formatted according to format.
func (jl *JSONLog) Format(format string) (string, error) {
	if format == "" {
		return jl.Log, nil
	}
	if format == "json" {
		m, err := json.Marshal(jl)
		return string(m), err
	}
	return fmt.Sprintf("%s %s", jl.Created.Format(format), jl.Log), nil
}

// Reset resets the log to nil.
func (jl *JSONLog) Reset() {
	jl.Log = ""
	jl.Stream = ""
	jl.Created = time.Time{}
}
