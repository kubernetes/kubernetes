// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

/*
High Performance, Feature-Rich Idiomatic Go codec/encoding library for 
binc, msgpack, cbor, json.

Supported Serialization formats are:

  - msgpack: https://github.com/msgpack/msgpack
  - binc:    http://github.com/ugorji/binc
  - cbor:    http://cbor.io http://tools.ietf.org/html/rfc7049
  - json:    http://json.org http://tools.ietf.org/html/rfc7159
  - simple: 

To install:

    go get github.com/ugorji/go/codec

This package understands the 'unsafe' tag, to allow using unsafe semantics:

  - When decoding into a struct, you need to read the field name as a string 
    so you can find the struct field it is mapped to.
    Using `unsafe` will bypass the allocation and copying overhead of []byte->string conversion.

To install using unsafe, pass the 'unsafe' tag:

    go get -tags=unsafe github.com/ugorji/go/codec

For detailed usage information, read the primer at http://ugorji.net/blog/go-codec-primer .

The idiomatic Go support is as seen in other encoding packages in
the standard library (ie json, xml, gob, etc).

Rich Feature Set includes:

  - Simple but extremely powerful and feature-rich API
  - Very High Performance.
    Our extensive benchmarks show us outperforming Gob, Json, Bson, etc by 2-4X.
  - Multiple conversions:
    Package coerces types where appropriate 
    e.g. decode an int in the stream into a float, etc.
  - Corner Cases: 
    Overflows, nil maps/slices, nil values in streams are handled correctly
  - Standard field renaming via tags
  - Support for omitting empty fields during an encoding
  - Encoding from any value and decoding into pointer to any value
    (struct, slice, map, primitives, pointers, interface{}, etc)
  - Extensions to support efficient encoding/decoding of any named types
  - Support encoding.(Binary|Text)(M|Unm)arshaler interfaces
  - Decoding without a schema (into a interface{}).
    Includes Options to configure what specific map or slice type to use
    when decoding an encoded list or map into a nil interface{}
  - Encode a struct as an array, and decode struct from an array in the data stream
  - Comprehensive support for anonymous fields
  - Fast (no-reflection) encoding/decoding of common maps and slices
  - Code-generation for faster performance.
  - Support binary (e.g. messagepack, cbor) and text (e.g. json) formats
  - Support indefinite-length formats to enable true streaming 
    (for formats which support it e.g. json, cbor)
  - Support canonical encoding, where a value is ALWAYS encoded as same sequence of bytes.
    This mostly applies to maps, where iteration order is non-deterministic.
  - NIL in data stream decoded as zero value
  - Never silently skip data when decoding.
    User decides whether to return an error or silently skip data when keys or indexes
    in the data stream do not map to fields in the struct.
  - Detect and error when encoding a cyclic reference (instead of stack overflow shutdown)
  - Encode/Decode from/to chan types (for iterative streaming support)
  - Drop-in replacement for encoding/json. `json:` key in struct tag supported.
  - Provides a RPC Server and Client Codec for net/rpc communication protocol.
  - Handle unique idiosynchracies of codecs e.g. 
    - For messagepack, configure how ambiguities in handling raw bytes are resolved 
    - For messagepack, provide rpc server/client codec to support 
      msgpack-rpc protocol defined at:
      https://github.com/msgpack-rpc/msgpack-rpc/blob/master/spec.md
  
Extension Support

Users can register a function to handle the encoding or decoding of
their custom types.

There are no restrictions on what the custom type can be. Some examples:

    type BisSet   []int
    type BitSet64 uint64
    type UUID     string
    type MyStructWithUnexportedFields struct { a int; b bool; c []int; }
    type GifImage struct { ... }

As an illustration, MyStructWithUnexportedFields would normally be
encoded as an empty map because it has no exported fields, while UUID
would be encoded as a string. However, with extension support, you can
encode any of these however you like.

RPC

RPC Client and Server Codecs are implemented, so the codecs can be used
with the standard net/rpc package.

Usage

The Handle is SAFE for concurrent READ, but NOT SAFE for concurrent modification.

The Encoder and Decoder are NOT safe for concurrent use.

Consequently, the usage model is basically:

    - Create and initialize the Handle before any use.
      Once created, DO NOT modify it.
    - Multiple Encoders or Decoders can now use the Handle concurrently.
      They only read information off the Handle (never write).
    - However, each Encoder or Decoder MUST not be used concurrently
    - To re-use an Encoder/Decoder, call Reset(...) on it first.
      This allows you use state maintained on the Encoder/Decoder.

Sample usage model:

    // create and configure Handle
    var (
      bh codec.BincHandle
      mh codec.MsgpackHandle
      ch codec.CborHandle
    )

    mh.MapType = reflect.TypeOf(map[string]interface{}(nil))

    // configure extensions
    // e.g. for msgpack, define functions and enable Time support for tag 1
    // mh.SetExt(reflect.TypeOf(time.Time{}), 1, myExt)

    // create and use decoder/encoder
    var (
      r io.Reader
      w io.Writer
      b []byte
      h = &bh // or mh to use msgpack
    )

    dec = codec.NewDecoder(r, h)
    dec = codec.NewDecoderBytes(b, h)
    err = dec.Decode(&v)

    enc = codec.NewEncoder(w, h)
    enc = codec.NewEncoderBytes(&b, h)
    err = enc.Encode(v)

    //RPC Server
    go func() {
        for {
            conn, err := listener.Accept()
            rpcCodec := codec.GoRpc.ServerCodec(conn, h)
            //OR rpcCodec := codec.MsgpackSpecRpc.ServerCodec(conn, h)
            rpc.ServeCodec(rpcCodec)
        }
    }()

    //RPC Communication (client side)
    conn, err = net.Dial("tcp", "localhost:5555")
    rpcCodec := codec.GoRpc.ClientCodec(conn, h)
    //OR rpcCodec := codec.MsgpackSpecRpc.ClientCodec(conn, h)
    client := rpc.NewClientWithCodec(rpcCodec)

*/
package codec

// Benefits of go-codec:
//
//    - encoding/json always reads whole file into memory first.
//      This makes it unsuitable for parsing very large files.
//    - encoding/xml cannot parse into a map[string]interface{}
//      I found this out on reading https://github.com/clbanning/mxj

// TODO:
//
//   - optimization for codecgen:
//     if len of entity is <= 3 words, then support a value receiver for encode.
//   - (En|De)coder should store an error when it occurs.
//     Until reset, subsequent calls return that error that was stored.
//     This means that free panics must go away.
//     All errors must be raised through errorf method.
//   - Decoding using a chan is good, but incurs concurrency costs.
//     This is because there's no fast way to use a channel without it
//     having to switch goroutines constantly.
//     Callback pattern is still the best. Maybe cnsider supporting something like:
//        type X struct {
//             Name string
//             Ys []Y
//             Ys chan <- Y
//             Ys func(Y) -> call this function for each entry
//        }
//    - Consider adding a isZeroer interface { isZero() bool }
//      It is used within isEmpty, for omitEmpty support.
//    - Consider making Handle used AS-IS within the encoding/decoding session.
//      This means that we don't cache Handle information within the (En|De)coder,
//      except we really need it at Reset(...)
//    - Consider adding math/big support
//    - Consider reducing the size of the generated functions:
//      Maybe use one loop, and put the conditionals in the loop.
//      for ... { if cLen > 0 { if j == cLen { break } } else if dd.CheckBreak() { break } }
