package formatter

import (
	"encoding/json"
	"os"

	"github.com/mgechev/revive/lint"
)

// NDJSON is an implementation of the Formatter interface
// which formats the errors to NDJSON stream.
type NDJSON struct {
	Metadata lint.FormatterMetadata
}

// Name returns the name of the formatter
func (f *NDJSON) Name() string {
	return "ndjson"
}

// Format formats the failures gotten from the lint.
func (f *NDJSON) Format(failures <-chan lint.Failure, config lint.Config) (string, error) {
	enc := json.NewEncoder(os.Stdout)
	for failure := range failures {
		obj := jsonObject{}
		obj.Severity = severity(config, failure)
		obj.Failure = failure
		err := enc.Encode(obj)
		if err != nil {
			return "", err
		}
	}
	return "", nil
}
