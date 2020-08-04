package jsoniter

import (
	"encoding/json"
	"github.com/modern-go/reflect2"
	"unsafe"
)

var jsonRawMessageType = reflect2.TypeOfPtr((*json.RawMessage)(nil)).Elem()
var jsoniterRawMessageType = reflect2.TypeOfPtr((*RawMessage)(nil)).Elem()

func createEncoderOfJsonRawMessage(ctx *ctx, typ reflect2.Type) ValEncoder {
	if typ == jsonRawMessageType {
		return &jsonRawMessageCodec{}
	}
	if typ == jsoniterRawMessageType {
		return &jsoniterRawMessageCodec{}
	}
	return nil
}

func createDecoderOfJsonRawMessage(ctx *ctx, typ reflect2.Type) ValDecoder {
	if typ == jsonRawMessageType {
		return &jsonRawMessageCodec{}
	}
	if typ == jsoniterRawMessageType {
		return &jsoniterRawMessageCodec{}
	}
	return nil
}

type jsonRawMessageCodec struct {
}

func (codec *jsonRawMessageCodec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	*((*json.RawMessage)(ptr)) = json.RawMessage(iter.SkipAndReturnBytes())
}

func (codec *jsonRawMessageCodec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteRaw(string(*((*json.RawMessage)(ptr))))
}

func (codec *jsonRawMessageCodec) IsEmpty(ptr unsafe.Pointer) bool {
	return len(*((*json.RawMessage)(ptr))) == 0
}

type jsoniterRawMessageCodec struct {
}

func (codec *jsoniterRawMessageCodec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	*((*RawMessage)(ptr)) = RawMessage(iter.SkipAndReturnBytes())
}

func (codec *jsoniterRawMessageCodec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteRaw(string(*((*RawMessage)(ptr))))
}

func (codec *jsoniterRawMessageCodec) IsEmpty(ptr unsafe.Pointer) bool {
	return len(*((*RawMessage)(ptr))) == 0
}
