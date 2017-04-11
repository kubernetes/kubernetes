package jose

import (
	"testing"
)

func TestDecodeBase64URLPaddingOptional(t *testing.T) {
	tests := []struct {
		encoded string
		decoded string
		err     bool
	}{
		{
			// With padding
			encoded: "VGVjdG9uaWM=",
			decoded: "Tectonic",
		},
		{
			// Without padding
			encoded: "VGVjdG9uaWM",
			decoded: "Tectonic",
		},
		{
			// Even More padding
			encoded: "VGVjdG9uaQ==",
			decoded: "Tectoni",
		},
		{
			// And take it away!
			encoded: "VGVjdG9uaQ",
			decoded: "Tectoni",
		},
		{
			// Too much padding.
			encoded: "VGVjdG9uaWNh=",
			decoded: "",
			err:     true,
		},
		{
			// Too much padding.
			encoded: "VGVjdG9uaWNh=",
			decoded: "",
			err:     true,
		},
	}

	for i, tt := range tests {
		got, err := decodeBase64URLPaddingOptional(tt.encoded)
		if tt.err {
			if err == nil {
				t.Errorf("case %d: expected non-nil err", i)
			}
			continue
		}

		if err != nil {
			t.Errorf("case %d: want nil err, got: %v", i, err)
		}

		if string(got) != tt.decoded {
			t.Errorf("case %d: want=%q, got=%q", i, tt.decoded, got)
		}
	}
}
