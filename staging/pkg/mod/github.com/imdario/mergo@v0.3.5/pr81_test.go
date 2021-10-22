package mergo

import (
	"testing"
)

func TestMapInterfaceWithMultipleLayer(t *testing.T) {
	m1 := map[string]interface{}{
		"k1": map[string]interface{}{
			"k1.1": "v1",
		},
	}

	m2 := map[string]interface{}{
		"k1": map[string]interface{}{
			"k1.1": "v2",
			"k1.2": "v3",
		},
	}

	if err := Map(&m1, m2, WithOverride); err != nil {
		t.Fatalf("Error merging: %v", err)
	}

	// Check overwrite of sub map works
	expected := "v2"
	actual := m1["k1"].(map[string]interface{})["k1.1"].(string)
	if actual != expected {
		t.Fatalf("Expected %v but got %v",
			expected,
			actual)
	}

	// Check new key is merged
	expected = "v3"
	actual = m1["k1"].(map[string]interface{})["k1.2"].(string)
	if actual != expected {
		t.Fatalf("Expected %v but got %v",
			expected,
			actual)
	}
}
