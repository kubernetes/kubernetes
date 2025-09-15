// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"sort"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/encoding/messageset"
	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/internal/flags"
)

func sizeMessageSet(mi *MessageInfo, p pointer, opts marshalOptions) (size int) {
	if !flags.ProtoLegacy {
		return 0
	}

	ext := *p.Apply(mi.extensionOffset).Extensions()
	for _, x := range ext {
		xi := getExtensionFieldInfo(x.Type())
		if xi.funcs.size == nil {
			continue
		}
		num, _ := protowire.DecodeTag(xi.wiretag)
		size += messageset.SizeField(num)
		if fullyLazyExtensions(opts) {
			// Don't expand the extension, instead use the buffer to calculate size
			if lb := x.lazyBuffer(); lb != nil {
				// We got hold of the buffer, so it's still lazy.
				// Don't count the tag size in the extension buffer, it's already added.
				size += protowire.SizeTag(messageset.FieldMessage) + len(lb) - xi.tagsize
				continue
			}
		}
		size += xi.funcs.size(x.Value(), protowire.SizeTag(messageset.FieldMessage), opts)
	}

	if u := mi.getUnknownBytes(p); u != nil {
		size += messageset.SizeUnknown(*u)
	}

	return size
}

func marshalMessageSet(mi *MessageInfo, b []byte, p pointer, opts marshalOptions) ([]byte, error) {
	if !flags.ProtoLegacy {
		return b, errors.New("no support for message_set_wire_format")
	}

	ext := *p.Apply(mi.extensionOffset).Extensions()
	switch len(ext) {
	case 0:
	case 1:
		// Fast-path for one extension: Don't bother sorting the keys.
		for _, x := range ext {
			var err error
			b, err = marshalMessageSetField(mi, b, x, opts)
			if err != nil {
				return b, err
			}
		}
	default:
		// Sort the keys to provide a deterministic encoding.
		// Not sure this is required, but the old code does it.
		keys := make([]int, 0, len(ext))
		for k := range ext {
			keys = append(keys, int(k))
		}
		sort.Ints(keys)
		for _, k := range keys {
			var err error
			b, err = marshalMessageSetField(mi, b, ext[int32(k)], opts)
			if err != nil {
				return b, err
			}
		}
	}

	if u := mi.getUnknownBytes(p); u != nil {
		var err error
		b, err = messageset.AppendUnknown(b, *u)
		if err != nil {
			return b, err
		}
	}

	return b, nil
}

func marshalMessageSetField(mi *MessageInfo, b []byte, x ExtensionField, opts marshalOptions) ([]byte, error) {
	xi := getExtensionFieldInfo(x.Type())
	num, _ := protowire.DecodeTag(xi.wiretag)
	b = messageset.AppendFieldStart(b, num)

	if fullyLazyExtensions(opts) {
		// Don't expand the extension if it's still in wire format, instead use the buffer content.
		if lb := x.lazyBuffer(); lb != nil {
			// The tag inside the lazy buffer is a different tag (the extension
			// number), but what we need here is the tag for FieldMessage:
			b = protowire.AppendVarint(b, protowire.EncodeTag(messageset.FieldMessage, protowire.BytesType))
			b = append(b, lb[xi.tagsize:]...)
			b = messageset.AppendFieldEnd(b)
			return b, nil
		}
	}

	b, err := xi.funcs.marshal(b, x.Value(), protowire.EncodeTag(messageset.FieldMessage, protowire.BytesType), opts)
	if err != nil {
		return b, err
	}
	b = messageset.AppendFieldEnd(b)
	return b, nil
}

func unmarshalMessageSet(mi *MessageInfo, b []byte, p pointer, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if !flags.ProtoLegacy {
		return out, errors.New("no support for message_set_wire_format")
	}

	ep := p.Apply(mi.extensionOffset).Extensions()
	if *ep == nil {
		*ep = make(map[int32]ExtensionField)
	}
	ext := *ep
	initialized := true
	err = messageset.Unmarshal(b, true, func(num protowire.Number, v []byte) error {
		o, err := mi.unmarshalExtension(v, num, protowire.BytesType, ext, opts)
		if err == errUnknown {
			u := mi.mutableUnknownBytes(p)
			*u = protowire.AppendTag(*u, num, protowire.BytesType)
			*u = append(*u, v...)
			return nil
		}
		if !o.initialized {
			initialized = false
		}
		return err
	})
	out.n = len(b)
	out.initialized = initialized
	return out, err
}
