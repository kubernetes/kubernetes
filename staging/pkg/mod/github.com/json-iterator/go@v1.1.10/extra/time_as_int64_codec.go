package extra

import (
	"github.com/json-iterator/go"
	"time"
	"unsafe"
)

// RegisterTimeAsInt64Codec encode/decode time since number of unit since epoch. the precision is the unit.
func RegisterTimeAsInt64Codec(precision time.Duration) {
	jsoniter.RegisterTypeEncoder("time.Time", &timeAsInt64Codec{precision})
	jsoniter.RegisterTypeDecoder("time.Time", &timeAsInt64Codec{precision})
}

type timeAsInt64Codec struct {
	precision time.Duration
}

func (codec *timeAsInt64Codec) Decode(ptr unsafe.Pointer, iter *jsoniter.Iterator) {
	nanoseconds := iter.ReadInt64() * codec.precision.Nanoseconds()
	*((*time.Time)(ptr)) = time.Unix(0, nanoseconds)
}

func (codec *timeAsInt64Codec) IsEmpty(ptr unsafe.Pointer) bool {
	ts := *((*time.Time)(ptr))
	return ts.UnixNano() == 0
}
func (codec *timeAsInt64Codec) Encode(ptr unsafe.Pointer, stream *jsoniter.Stream) {
	ts := *((*time.Time)(ptr))
	stream.WriteInt64(ts.UnixNano() / codec.precision.Nanoseconds())
}
