package imperativeevictionresponder

import (
	"math"
	"testing"
)

func TestGetRecordedAttempts(t *testing.T) {
	tests := []struct {
		name    string
		message string
		want    uint64
	}{
		{
			name:    "missing message",
			message: "",
			want:    0,
		},
		{
			name:    "missing attempts",
			message: "foo pod (uid=\"492fe0ed-e8ee-4ad4-900f-591e35d6d352\") deletion",
			want:    0,
		},
		{
			name:    "invalid attempts",
			message: "foo pod (uid=\"492fe0ed-e8ee-4ad4-900f-591e35d6d352\") deletion via the /eviction subresource failed (attempts=bar): pods \"foo\" is forbidden:",
			want:    0,
		},
		{
			name:    "negative number attempts",
			message: "foo pod (uid=\"492fe0ed-e8ee-4ad4-900f-591e35d6d352\") deletion via the /eviction subresource failed (attempts=-7): pods \"foo\" is forbidden:",
			want:    0,
		},
		{
			name:    "found 0",
			message: "foo pod (uid=\"492fe0ed-e8ee-4ad4-900f-591e35d6d352\") deletion via the /eviction subresource failed (attempts=0): pods \"foo\" is forbidden:",
			want:    0,
		},
		{
			name:    "found 1",
			message: "foo pod (uid=\"492fe0ed-e8ee-4ad4-900f-591e35d6d352\") deletion via the /eviction subresource failed (attempts=1): pods \"foo\" is forbidden:",
			want:    1,
		},
		{
			name:    "found 7 with preceding zeros",
			message: "foo pod (uid=\"492fe0ed-e8ee-4ad4-900f-591e35d6d352\") deletion via the /eviction subresource failed (attempts=007): pods \"foo\" is forbidden:",
			want:    7,
		},
		{
			name:    "found MaxUint64",
			message: "foo pod (uid=\"492fe0ed-e8ee-4ad4-900f-591e35d6d352\") deletion via the /eviction subresource failed (attempts=18446744073709551615): pods \"foo\" is forbidden:",
			want:    math.MaxUint64,
		},
		{
			name:    "MaxUint64 overflow",
			message: "foo pod (uid=\"492fe0ed-e8ee-4ad4-900f-591e35d6d352\") deletion via the /eviction subresource failed (attempts=18446744073709551616): pods \"foo\" is forbidden:",
			want:    0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			attempts := getRecordedAttempts(tc.message)
			if attempts != tc.want {
				t.Errorf("expected %d, got %d", tc.want, attempts)
			}
		})
	}
}
