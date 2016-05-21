// +build appengine

package msgpack

import (
	"reflect"

	ds "google.golang.org/appengine/datastore"
)

var (
	keyPtrType = reflect.TypeOf((*ds.Key)(nil))
	cursorType = reflect.TypeOf((*ds.Cursor)(nil)).Elem()
)

func init() {
	Register(keyPtrType, encodeDatastoreKeyValue, decodeDatastoreKeyValue)
	Register(cursorType, encodeDatastoreCursorValue, decodeDatastoreCursorValue)
}

func EncodeDatastoreKey(e *Encoder, key *ds.Key) error {
	if key == nil {
		return e.EncodeNil()
	}
	return e.EncodeString(key.Encode())
}

func encodeDatastoreKeyValue(e *Encoder, v reflect.Value) error {
	key := v.Interface().(*ds.Key)
	return EncodeDatastoreKey(e, key)
}

func DecodeDatastoreKey(d *Decoder) (*ds.Key, error) {
	v, err := d.DecodeString()
	if err != nil {
		return nil, err
	}
	if v == "" {
		return nil, nil
	}
	return ds.DecodeKey(v)
}

func decodeDatastoreKeyValue(d *Decoder, v reflect.Value) error {
	key, err := DecodeDatastoreKey(d)
	if err != nil {
		return err
	}
	v.Set(reflect.ValueOf(key))
	return nil
}

func encodeDatastoreCursorValue(e *Encoder, v reflect.Value) error {
	cursor := v.Interface().(ds.Cursor)
	return e.Encode(cursor.String())
}

func decodeDatastoreCursorValue(d *Decoder, v reflect.Value) error {
	s, err := d.DecodeString()
	if err != nil {
		return err
	}
	cursor, err := ds.DecodeCursor(s)
	if err != nil {
		return err
	}
	v.Set(reflect.ValueOf(cursor))
	return nil
}
