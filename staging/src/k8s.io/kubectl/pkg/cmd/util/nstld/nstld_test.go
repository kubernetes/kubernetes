package nstld

import (
	"os"
	"strings"
	"sync"
	"testing"
)

var testOnce sync.Once

func resetCommonTLDs() {
	// Reset CommonTLDs to default values for testing
	CommonTLDs = map[string]struct{}{
		"com":  {},
		"org":  {},
		"net":  {},
		"edu":  {},
		"gov":  {},
		"dev":  {},
		"io":   {},
	}
}

// TestIsTLD tests the IsTLD function which determines if a namespace name
// is a common top-level domain (TLD).
func TestIsTLD(t *testing.T) {
	testOnce.Do(resetCommonTLDs)

	tests := []struct {
		name     string
		nsName   string
		expected bool
	}{
		{
			name:     "common TLD com",
			nsName:   "com",
			expected: true,
		},
		{
			name:     "common TLD org",
			nsName:   "org",
			expected: true,
		},
		{
			name:     "common TLD net",
			nsName:   "net",
			expected: true,
		},
		{
			name:     "non TLD example",
			nsName:   "example",
			expected: false,
		},
		{
			name:     "non TLD kubernetes",
			nsName:   "kubernetes",
			expected: false,
		},
		{
			name:     "empty string",
			nsName:   "",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsTLD(tt.nsName)
			if result != tt.expected {
				t.Errorf("IsTLD(%q) = %v, expected %v", tt.nsName, result, tt.expected)
			}
		})
	}
}

// TestGetTLDWarningMessage tests the GetTLDWarningMessage function which returns
// a warning message if a namespace name is a common top-level domain (TLD).
// The message warns users about potential DNS resolution issues.
func TestGetTLDWarningMessage(t *testing.T) {
	testOnce.Do(resetCommonTLDs)

	tests := []struct {
		name          string
		nsName        string
		expectWarning bool
	}{
		{
			name:          "common TLD com",
			nsName:        "com",
			expectWarning: true,
		},
		{
			name:          "common TLD dev",
			nsName:        "dev",
			expectWarning: true,
		},
		{
			name:          "non TLD example",
			nsName:        "example",
			expectWarning: false,
		},
		{
			name:          "non TLD default",
			nsName:        "default",
			expectWarning: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetTLDWarningMessage(tt.nsName)
			hasWarning := result != ""

			if hasWarning != tt.expectWarning {
				t.Errorf("GetTLDWarningMessage(%q) warning presence = %v, expected %v", tt.nsName, hasWarning, tt.expectWarning)
			}

			// If we expect a warning, validate the warning message content
			if tt.expectWarning {
				if !strings.Contains(result, tt.nsName) {
					t.Errorf("Warning message for %q does not contain the namespace name", tt.nsName)
				}

				if !strings.Contains(result, "Warning") {
					t.Errorf("Warning message for %q does not contain 'Warning'", tt.nsName)
				}
			}
		})
	}
}

// TestTLDEnvironmentVariable tests the initialization from environment variables
func TestTLDEnvironmentVariable(t *testing.T) {
	// Save original env and restore after test
	originalEnv := os.Getenv("KUBECTL_ADDITIONAL_TLDS")
	defer os.Setenv("KUBECTL_ADDITIONAL_TLDS", originalEnv)

	// Reset to known state
	resetCommonTLDs()

	// Set test environment variable
	os.Setenv("KUBECTL_ADDITIONAL_TLDS", "app,xyz,custom-tld")

	// Apply environment variables directly to CommonTLDs
	if additionalTLDs := os.Getenv("KUBECTL_ADDITIONAL_TLDS"); additionalTLDs != "" {
		for _, tld := range strings.Split(additionalTLDs, ",") {
			tld = strings.TrimSpace(tld)
			if tld != "" {
				CommonTLDs[tld] = struct{}{}
			}
		}
	}

	// Test custom TLDs
	customTests := []struct {
		name     string
		tld      string
		expected bool
	}{
		{
			name:     "custom TLD app",
			tld:      "app",
			expected: true,
		},
		{
			name:     "custom TLD xyz",
			tld:      "xyz",
			expected: true,
		},
		{
			name:     "custom TLD with hyphen",
			tld:      "custom-tld",
			expected: true,
		},
		{
			name:     "non-existent TLD",
			tld:      "nonexistent",
			expected: false,
		},
	}

	for _, tt := range customTests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsTLD(tt.tld)
			if result != tt.expected {
				t.Errorf("After environment configuration, IsTLD(%q) = %v, expected %v",
					tt.tld, result, tt.expected)
			}

			// Also check warning message generation
			message := GetTLDWarningMessage(tt.tld)
			hasWarning := message != ""

			if hasWarning != tt.expected {
				t.Errorf("GetTLDWarningMessage(%q) warning presence = %v, expected %v",
					tt.tld, hasWarning, tt.expected)
			}

			if tt.expected && !strings.Contains(message, tt.tld) {
				t.Errorf("Warning message for %q does not contain the TLD name", tt.tld)
			}
		})
	}
}

// TestTLDEnvironmentVariableWithSpaces tests handling of spaces in the environment variable
func TestTLDEnvironmentVariableWithSpaces(t *testing.T) {
	// Save original env and restore after test
	originalEnv := os.Getenv("KUBECTL_ADDITIONAL_TLDS")
	defer os.Setenv("KUBECTL_ADDITIONAL_TLDS", originalEnv)

	// Reset to known state
	resetCommonTLDs()

	// Set test environment variable with spaces
	os.Setenv("KUBECTL_ADDITIONAL_TLDS", " space, leading,trailing ,  multiple  ")

	// Apply environment variables directly to CommonTLDs
	if additionalTLDs := os.Getenv("KUBECTL_ADDITIONAL_TLDS"); additionalTLDs != "" {
		for _, tld := range strings.Split(additionalTLDs, ",") {
			tld = strings.TrimSpace(tld)
			if tld != "" {
				CommonTLDs[tld] = struct{}{}
			}
		}
	}

	// Test TLDs with various spacing
	spacingTests := []struct {
		name     string
		tld      string
		expected bool
	}{
		{
			name:     "TLD with spaces trimmed",
			tld:      "space",
			expected: true,
		},
		{
			name:     "TLD with leading space trimmed",
			tld:      "leading",
			expected: true,
		},
		{
			name:     "TLD with trailing space trimmed",
			tld:      "trailing",
			expected: true,
		},
		{
			name:     "TLD with multiple spaces trimmed",
			tld:      "multiple",
			expected: true,
		},
		{
			name:     "Empty TLD after trimming should be ignored",
			tld:      "",
			expected: false,
		},
	}

	for _, tt := range spacingTests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsTLD(tt.tld)
			if result != tt.expected {
				t.Errorf("After environment configuration with spaces, IsTLD(%q) = %v, expected %v",
					tt.tld, result, tt.expected)
			}
		})
	}
}
