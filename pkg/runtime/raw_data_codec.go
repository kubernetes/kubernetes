package runtime

import (
	"fmt"
)

func NewRawDataCodec(input Codec) (Codec, error) {
	wrapper, ok := input.(*codecWrapper)
	if !ok {
		return nil, fmt.Errorf("unexpected codec: %v", input)
	}
	scheme, ok := wrapper.ObjectCodec.(*Scheme)
	if !ok {
		return nil, fmt.Errorf("unexpected object codec: %v", wrapper.ObjectCodec)
	}
	// Copy it so we don't mess with other things...
	newRaw := *scheme.Raw()
	newRaw.SaveRawData = true
	
	newScheme := Scheme{
		raw: &newRaw,
		fieldLabelConversionFuncs: scheme.fieldLabelConversionFuncs,
	}
	
	return CodecFor(&newScheme, wrapper.version), nil
}
