// Package streamformatter provides helper functions to format a stream.
package streamformatter

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/docker/docker/pkg/jsonmessage"
	"github.com/docker/docker/pkg/progress"
)

const streamNewline = "\r\n"

type jsonProgressFormatter struct{}

func appendNewline(source []byte) []byte {
	return append(source, []byte(streamNewline)...)
}

// FormatStatus formats the specified objects according to the specified format (and id).
func FormatStatus(id, format string, a ...interface{}) []byte {
	str := fmt.Sprintf(format, a...)
	b, err := json.Marshal(&jsonmessage.JSONMessage{ID: id, Status: str})
	if err != nil {
		return FormatError(err)
	}
	return appendNewline(b)
}

// FormatError formats the error as a JSON object
func FormatError(err error) []byte {
	jsonError, ok := err.(*jsonmessage.JSONError)
	if !ok {
		jsonError = &jsonmessage.JSONError{Message: err.Error()}
	}
	if b, err := json.Marshal(&jsonmessage.JSONMessage{Error: jsonError, ErrorMessage: err.Error()}); err == nil {
		return appendNewline(b)
	}
	return []byte(`{"error":"format error"}` + streamNewline)
}

func (sf *jsonProgressFormatter) formatStatus(id, format string, a ...interface{}) []byte {
	return FormatStatus(id, format, a...)
}

// formatProgress formats the progress information for a specified action.
func (sf *jsonProgressFormatter) formatProgress(id, action string, progress *jsonmessage.JSONProgress, aux interface{}) []byte {
	if progress == nil {
		progress = &jsonmessage.JSONProgress{}
	}
	var auxJSON *json.RawMessage
	if aux != nil {
		auxJSONBytes, err := json.Marshal(aux)
		if err != nil {
			return nil
		}
		auxJSON = new(json.RawMessage)
		*auxJSON = auxJSONBytes
	}
	b, err := json.Marshal(&jsonmessage.JSONMessage{
		Status:          action,
		ProgressMessage: progress.String(),
		Progress:        progress,
		ID:              id,
		Aux:             auxJSON,
	})
	if err != nil {
		return nil
	}
	return appendNewline(b)
}

type rawProgressFormatter struct{}

func (sf *rawProgressFormatter) formatStatus(id, format string, a ...interface{}) []byte {
	return []byte(fmt.Sprintf(format, a...) + streamNewline)
}

func (sf *rawProgressFormatter) formatProgress(id, action string, progress *jsonmessage.JSONProgress, aux interface{}) []byte {
	if progress == nil {
		progress = &jsonmessage.JSONProgress{}
	}
	endl := "\r"
	if progress.String() == "" {
		endl += "\n"
	}
	return []byte(action + " " + progress.String() + endl)
}

// NewProgressOutput returns a progress.Output object that can be passed to
// progress.NewProgressReader.
func NewProgressOutput(out io.Writer) progress.Output {
	return &progressOutput{sf: &rawProgressFormatter{}, out: out, newLines: true}
}

// NewJSONProgressOutput returns a progress.Output that that formats output
// using JSON objects
func NewJSONProgressOutput(out io.Writer, newLines bool) progress.Output {
	return &progressOutput{sf: &jsonProgressFormatter{}, out: out, newLines: newLines}
}

type formatProgress interface {
	formatStatus(id, format string, a ...interface{}) []byte
	formatProgress(id, action string, progress *jsonmessage.JSONProgress, aux interface{}) []byte
}

type progressOutput struct {
	sf       formatProgress
	out      io.Writer
	newLines bool
}

// WriteProgress formats progress information from a ProgressReader.
func (out *progressOutput) WriteProgress(prog progress.Progress) error {
	var formatted []byte
	if prog.Message != "" {
		formatted = out.sf.formatStatus(prog.ID, prog.Message)
	} else {
		jsonProgress := jsonmessage.JSONProgress{Current: prog.Current, Total: prog.Total, HideCounts: prog.HideCounts, Units: prog.Units}
		formatted = out.sf.formatProgress(prog.ID, prog.Action, &jsonProgress, prog.Aux)
	}
	_, err := out.out.Write(formatted)
	if err != nil {
		return err
	}

	if out.newLines && prog.LastUpdate {
		_, err = out.out.Write(out.sf.formatStatus("", ""))
		return err
	}

	return nil
}

// AuxFormatter is a streamFormatter that writes aux progress messages
type AuxFormatter struct {
	io.Writer
}

// Emit emits the given interface as an aux progress message
func (sf *AuxFormatter) Emit(aux interface{}) error {
	auxJSONBytes, err := json.Marshal(aux)
	if err != nil {
		return err
	}
	auxJSON := new(json.RawMessage)
	*auxJSON = auxJSONBytes
	msgJSON, err := json.Marshal(&jsonmessage.JSONMessage{Aux: auxJSON})
	if err != nil {
		return err
	}
	msgJSON = appendNewline(msgJSON)
	n, err := sf.Writer.Write(msgJSON)
	if n != len(msgJSON) {
		return io.ErrShortWrite
	}
	return err
}
