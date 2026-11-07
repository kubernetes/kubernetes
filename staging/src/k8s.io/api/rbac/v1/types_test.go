package v1

import (
	"encoding/json"
	"testing"
)

// TestRulesJSONSerialization verifies that nil and empty Rules slices
// are omitted from JSON serialization, while populated Rules are preserved.
// This behavior is relevant to issue #143163, where explicitly including
// a nil rules field in a Server-Side Apply configuration can contribute
// to field ownership conflicts with the RBAC aggregation controller.
func TestRulesJSONSerialization(t *testing.T) {
	tests := []struct {
		name        string
		obj         any
		wantOmitted bool // true = field should be omitted from JSON (nil/empty slice)
	}{
		// nil slice (default zero value)
		{"Role/nil", Role{}, true},
		// empty slice (explicitly set but no rules)
		{"Role/empty", Role{Rules: []PolicyRule{}}, true},
		// populated slice (has actual rules)
		{"Role/populated", Role{Rules: []PolicyRule{{Verbs: []string{"get"}}}}, false},
		// same cases for ClusterRole
		{"ClusterRole/nil", ClusterRole{}, true},
		{"ClusterRole/empty", ClusterRole{Rules: []PolicyRule{}}, true},
		{"ClusterRole/populated", ClusterRole{Rules: []PolicyRule{{Verbs: []string{"get"}}}}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Marshal the object to JSON
			data, err := json.Marshal(tt.obj)
			if err != nil {
				t.Fatalf("json.Marshal() error: %v", err)
			}

			// Parse JSON to check field presence
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(data, &fields); err != nil {
				t.Fatalf("json.Unmarshal() error: %v", err)
			}

			// Check if "rules" field exists in the JSON output
			_, exists := fields["rules"]
			// wantOmitted=true means field should NOT exist (exists=false)
			// wantOmitted=false means field SHOULD exist (exists=true)
			if exists != !tt.wantOmitted {
				t.Fatalf("unexpected rules field presence: wantOmitted=%v, exists=%v, JSON: %s",
					tt.wantOmitted, exists, data)
			}

			// For populated cases, verify the rules data is correctly preserved
			if !tt.wantOmitted {
				var rules []PolicyRule
				if err := json.Unmarshal(fields["rules"], &rules); err != nil {
					t.Fatalf("invalid rules JSON: %v", err)
				}
				if len(rules) != 1 || len(rules[0].Verbs) != 1 || rules[0].Verbs[0] != "get" {
					t.Fatalf("populated rules were not preserved: %s", data)
				}
			}
		})
	}
}

// TestRulesJSONUnmarshaling verifies backward compatibility: missing field,
// explicit null, and empty array all unmarshal to nil/empty slice in Go.
// This ensures existing clients and stored objects continue to work correctly.
func TestRulesJSONUnmarshaling(t *testing.T) {
	inputs := []string{
		`{}`,              // missing field
		`{"rules": null}`, // explicit null
		`{"rules": []}`,   // empty array
	}

	for _, input := range inputs {
		t.Run(input, func(t *testing.T) {
			var role Role
			if err := json.Unmarshal([]byte(input), &role); err != nil {
				t.Fatalf("unmarshal failed: %v", err)
			}
			if role.Rules != nil && len(role.Rules) != 0 {
				t.Fatalf("expected nil/empty Rules, got %v", role.Rules)
			}
		})
	}
}

// TestRulesRoundTrip verifies that marshaling and unmarshaling preserves data.
func TestRulesRoundTrip(t *testing.T) {
	original := ClusterRole{Rules: []PolicyRule{{Verbs: []string{"get", "list"}, APIGroups: []string{""}}}}
	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	var restored ClusterRole
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	if len(restored.Rules) != 1 || len(restored.Rules[0].Verbs) != 2 {
		t.Fatalf("round-trip failed: original=%v, restored=%v", original.Rules, restored.Rules)
	}
}
