package types

import "testing"

func TestNewUUID(t *testing.T) {
	tests := []struct {
		in string
		ws string
	}{
		{
			"6733C088-A507-4694-AABF-EDBE4FC5266F",

			"6733c088-a507-4694-aabf-edbe4fc5266f",
		},
		{
			"6733C088A5074694AABFEDBE4FC5266F",

			"6733c088-a507-4694-aabf-edbe4fc5266f",
		},
		{
			"0aaf0a79-1a39-4d59-abbf-1bebca8209d2",

			"0aaf0a79-1a39-4d59-abbf-1bebca8209d2",
		},
		{
			"0aaf0a791a394d59abbf1bebca8209d2",

			"0aaf0a79-1a39-4d59-abbf-1bebca8209d2",
		},
	}
	for i, tt := range tests {
		gu, err := NewUUID(tt.in)
		if err != nil {
			t.Errorf("#%d: err=%v, want %v", i, err, nil)
		}
		if gs := gu.String(); gs != tt.ws {
			t.Errorf("#%d: String()=%v, want %v", i, gs, tt.ws)
		}
	}
}

func TestNewUUIDBad(t *testing.T) {
	tests := []string{
		"asdf",
		"0AAF0A79-1A39-4D59-ABBF-1BEBCA8209D2ABC",
		"",
	}
	for i, tt := range tests {
		g, err := NewUUID(tt)
		if err == nil {
			t.Errorf("#%d: err=nil, want non-nil", i)
		}
		if g != nil {
			t.Errorf("#%d: err=%v, want %v", i, g, nil)

		}
	}

}
