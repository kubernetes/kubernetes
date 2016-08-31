package metrics

import "testing"

func TestCleanUserAgent(t *testing.T) {
	for _, tc := range []struct {
		In  string
		Out string
	}{
		{
			In:  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36",
			Out: "Mozilla",
		},
		{
			In:  "kubectl/v1.2.4",
			Out: "kubectl/v1.2.4",
		},
	} {
		if cleanUserAgent(tc.In) != tc.Out {
			t.Errorf("Failed to clean User-Agent: %s", tc.In)
		}
	}
}
