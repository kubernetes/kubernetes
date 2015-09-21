//+build x,codecgen

package codec

import (
	"fmt"
	"testing"
)

func TestCodecgenJson1(t *testing.T) {
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
