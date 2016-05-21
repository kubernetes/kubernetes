package msgpack

import (
	"reflect"

	"gopkg.in/vmihailenco/msgpack.v2/codes"
)

func (e *Encoder) EncodeMapLen(l int) error {
	if l < 16 {
		return e.w.WriteByte(codes.FixedMapLow | byte(l))
	}
	if l < 65536 {
		return e.write2(codes.Map16, uint64(l))
	}
	return e.write4(codes.Map32, uint64(l))
}

func (e *Encoder) encodeMapStringString(m map[string]string) error {
	if m == nil {
		return e.EncodeNil()
	}
	if err := e.EncodeMapLen(len(m)); err != nil {
		return err
	}
	for mk, mv := range m {
		if err := e.EncodeString(mk); err != nil {
			return err
		}
		if err := e.EncodeString(mv); err != nil {
			return err
		}
	}
	return nil
}

func (e *Encoder) encodeMap(v reflect.Value) error {
	if v.IsNil() {
		return e.EncodeNil()
	}
	if err := e.EncodeMapLen(v.Len()); err != nil {
		return err
	}
	for _, key := range v.MapKeys() {
		if err := e.EncodeValue(key); err != nil {
			return err
		}
		if err := e.EncodeValue(v.MapIndex(key)); err != nil {
			return err
		}
	}
	return nil
}

func (e *Encoder) encodeStruct(strct reflect.Value) error {
	structFields := structs.Fields(strct.Type())
	fields := structFields.OmitEmpty(strct)

	if err := e.EncodeMapLen(len(fields)); err != nil {
		return err
	}

	for _, f := range fields {
		if err := e.EncodeString(f.name); err != nil {
			return err
		}
		if err := f.EncodeValue(e, strct); err != nil {
			return err
		}
	}

	return nil
}
