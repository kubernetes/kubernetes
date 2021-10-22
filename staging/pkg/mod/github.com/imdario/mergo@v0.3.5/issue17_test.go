package mergo

import (
	"encoding/json"
	"testing"
)

var (
	request    = `{"timestamp":null, "name": "foo"}`
	maprequest = map[string]interface{}{
		"timestamp": nil,
		"name":      "foo",
		"newStuff":  "foo",
	}
)

func TestIssue17MergeWithOverwrite(t *testing.T) {
	var something map[string]interface{}
	if err := json.Unmarshal([]byte(request), &something); err != nil {
		t.Errorf("Error while Unmarshalling maprequest: %s", err)
	}
	if err := MergeWithOverwrite(&something, maprequest); err != nil {
		t.Errorf("Error while merging: %s", err)
	}
}
