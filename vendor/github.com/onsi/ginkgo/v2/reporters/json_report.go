package reporters

import (
	"encoding/json"
	"fmt"
	"os"
	"path"

	"github.com/onsi/ginkgo/v2/types"
)

// GenerateJSONReport produces a JSON-formatted report at the passed in destination
func GenerateJSONReport(report types.Report, destination string) error {
	if err := os.MkdirAll(path.Dir(destination), 0770); err != nil {
		return err
	}
	f, err := os.Create(destination)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	err = enc.Encode([]types.Report{
		report,
	})
	if err != nil {
		return err
	}
	return f.Close()
}

// MergeJSONReports produces a single JSON-formatted report at the passed in destination by merging the JSON-formatted reports provided in sources
// It skips over reports that fail to decode but reports on them via the returned messages []string
func MergeAndCleanupJSONReports(sources []string, destination string) ([]string, error) {
	messages := []string{}
	allReports := []types.Report{}
	for _, source := range sources {
		reports := []types.Report{}
		data, err := os.ReadFile(source)
		if err != nil {
			messages = append(messages, fmt.Sprintf("Could not open %s:\n%s", source, err.Error()))
			continue
		}
		err = json.Unmarshal(data, &reports)
		if err != nil {
			messages = append(messages, fmt.Sprintf("Could not decode %s:\n%s", source, err.Error()))
			continue
		}
		os.Remove(source)
		allReports = append(allReports, reports...)
	}

	if err := os.MkdirAll(path.Dir(destination), 0770); err != nil {
		return messages, err
	}
	f, err := os.Create(destination)
	if err != nil {
		return messages, err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	err = enc.Encode(allReports)
	if err != nil {
		return messages, err
	}
	return messages, f.Close()
}
