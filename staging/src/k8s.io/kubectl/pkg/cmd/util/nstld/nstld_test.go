package nstld

import (
    "strings"
    "testing"
)

// TestIsTLD tests the IsTLD function which determines if a namespace name
// is a common top-level domain (TLD).
func TestIsTLD(t *testing.T) {
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