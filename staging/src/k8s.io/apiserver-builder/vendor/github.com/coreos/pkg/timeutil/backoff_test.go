package timeutil

import (
	"testing"
	"time"
)

func TestExpBackoff(t *testing.T) {
	tests := []struct {
		prev time.Duration
		max  time.Duration
		want time.Duration
	}{
		{
			prev: time.Duration(0),
			max:  time.Minute,
			want: time.Second,
		},
		{
			prev: time.Second,
			max:  time.Minute,
			want: 2 * time.Second,
		},
		{
			prev: 16 * time.Second,
			max:  time.Minute,
			want: 32 * time.Second,
		},
		{
			prev: 32 * time.Second,
			max:  time.Minute,
			want: time.Minute,
		},
		{
			prev: time.Minute,
			max:  time.Minute,
			want: time.Minute,
		},
		{
			prev: 2 * time.Minute,
			max:  time.Minute,
			want: time.Minute,
		},
	}

	for i, tt := range tests {
		got := ExpBackoff(tt.prev, tt.max)
		if tt.want != got {
			t.Errorf("case %d: want=%v got=%v", i, tt.want, got)
		}
	}
}
