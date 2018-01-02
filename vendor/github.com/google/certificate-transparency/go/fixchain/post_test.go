package fixchain

import (
	"net/http"
	"testing"
)

func TestPostChainToLog(t *testing.T) {
	for i, test := range postTests {
		client := &http.Client{Transport: &postTestRoundTripper{t: t, test: &test, testIndex: i}}
		ferr := PostChainToLog(extractTestChain(t, i, test.chain), client, test.url)

		if ferr == nil {
			if test.ferr.Type != None {
				t.Errorf("#%d: PostChainToLog() didn't return FixError, expected FixError of type %s", i, test.ferr.TypeString())
			}
		} else {
			if ferr.Type != test.ferr.Type {
				t.Errorf("#%d: PostChainToLog() returned FixError of type %s, expected %s", i, ferr.TypeString(), test.ferr.TypeString())
			}
		}
	}
}
