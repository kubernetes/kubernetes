package garif

import (
	"encoding/json"
	"io"
)

// Write writes the JSON
func (l *LogFile) Write(w io.Writer) error {
	marshal, err := json.Marshal(l)
	if err != nil {
		return err
	}
	_, err = w.Write(marshal)
	return err
}

// PrettyWrite writes indented JSON
func (l *LogFile) PrettyWrite(w io.Writer) error {
	marshal, err := json.MarshalIndent(l, "", "  ")
	if err != nil {
		return err
	}
	_, err = w.Write(marshal)
	return err
}
