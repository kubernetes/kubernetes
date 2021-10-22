package testing

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/baremetalintrospection/v1/introspection"
)

func TestLLDPTLVErrors(t *testing.T) {
	badInputs := []string{
		"[1]",
		"[1, 2]",
		"[\"foo\", \"bar\"]",
	}

	for _, input := range badInputs {
		var output introspection.LLDPTLVType
		err := json.Unmarshal([]byte(input), &output)
		if err == nil {
			t.Errorf("No JSON parse error for invalid LLDP TLV %s", input)
		}
		if !strings.Contains(err.Error(), "LLDP TLV") {
			t.Errorf("Unexpected JSON parse error \"%s\" for invalid LLDP TLV %s",
				err, input)
		}
	}
}
