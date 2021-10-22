// +build go1.7

package eventstreamtest

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/aws/aws-sdk-go/private/protocol/eventstream"
)

// AssertMessageEqual compares to event stream messages, and determines if they
// are equal. Will trigger an testing Error if components of the message are
// not equal.
func AssertMessageEqual(t testing.TB, a, b eventstream.Message, msg ...interface{}) {
	getHelper(t)()

	ah, err := bytesEncodeHeader(a.Headers)
	if err != nil {
		t.Fatalf("unable to encode a's headers, %v", err)
	}

	bh, err := bytesEncodeHeader(b.Headers)
	if err != nil {
		t.Fatalf("unable to encode b's headers, %v", err)
	}

	if !bytes.Equal(ah, bh) {
		aj, err := json.Marshal(ah)
		if err != nil {
			t.Fatalf("unable to json encode a's headers, %v", err)
		}
		bj, err := json.Marshal(bh)
		if err != nil {
			t.Fatalf("unable to json encode b's headers, %v", err)
		}
		t.Errorf("%s\nexpect headers: %v\n\t%v\nactual headers: %v\n\t%v\n",
			fmt.Sprint(msg...),
			base64.StdEncoding.EncodeToString(ah), aj,
			base64.StdEncoding.EncodeToString(bh), bj,
		)
	}

	if !bytes.Equal(a.Payload, b.Payload) {
		t.Errorf("%s\nexpect payload: %v\nactual payload: %v\n",
			fmt.Sprint(msg...),
			base64.StdEncoding.EncodeToString(a.Payload),
			base64.StdEncoding.EncodeToString(b.Payload),
		)
	}
}

func bytesEncodeHeader(v eventstream.Headers) ([]byte, error) {
	var buf bytes.Buffer
	if err := eventstream.EncodeHeaders(&buf, v); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}
