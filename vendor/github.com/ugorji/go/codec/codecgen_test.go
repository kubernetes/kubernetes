//+build x,codecgen

package codec

import (
	"fmt"
	"testing"
)

func _TestCodecgenJson1(t *testing.T) {
	// This is just a simplistic test for codecgen.
	// It is typically disabled. We only enable it for debugging purposes.
	const callCodecgenDirect bool = true
	v := newTestStruc(2, false, !testSkipIntf, false)
	var bs []byte
	e := NewEncoderBytes(&bs, testJsonH)
	if callCodecgenDirect {
		v.CodecEncodeSelf(e)
		e.w.atEndOfEncode()
	} else {
		e.MustEncode(v)
	}
	fmt.Printf("%s\n", bs)
}
