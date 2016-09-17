# Codec

High Performance and Feature-Rich Idiomatic Go Library providing
encode/decode support for different serialization formats.

Supported Serialization formats are:

  - msgpack: [https://github.com/msgpack/msgpack]
  - binc: [http://github.com/ugorji/binc]

To install:

    go get github.com/ugorji/go/codec

Online documentation: [http://godoc.org/github.com/ugorji/go/codec]

The idiomatic Go support is as seen in other encoding packages in
the standard library (ie json, xml, gob, etc).

Rich Feature Set includes:

  - Simple but extremely powerful and feature-rich API
  - Very High Performance.   
    Our extensive benchmarks show us outperforming Gob, Json and Bson by 2-4X.
    This was achieved by taking extreme care on:
      - managing allocation
      - function frame size (important due to Go's use of split stacks),
      - reflection use (and by-passing reflection for common types)
      - recursion implications
      - zero-copy mode (encoding/decoding to byte slice without using temp buffers)
  - Correct.  
    Care was taken to precisely handle corner cases like: 
      overflows, nil maps and slices, nil value in stream, etc.
  - Efficient zero-copying into temporary byte buffers  
    when encoding into or decoding from a byte slice.
  - Standard field renaming via tags
  - Encoding from any value  
    (struct, slice, map, primitives, pointers, interface{}, etc)
  - Decoding into pointer to any non-nil typed value  
    (struct, slice, map, int, float32, bool, string, reflect.Value, etc)
  - Supports extension functions to handle the encode/decode of custom types
  - Support Go 1.2 encoding.BinaryMarshaler/BinaryUnmarshaler
  - Schema-less decoding  
    (decode into a pointer to a nil interface{} as opposed to a typed non-nil value).  
    Includes Options to configure what specific map or slice type to use 
    when decoding an encoded list or map into a nil interface{}
  - Provides a RPC Server and Client Codec for net/rpc communication protocol.
  - Msgpack Specific:
      - Provides extension functions to handle spec-defined extensions (binary, timestamp)
      - Options to resolve ambiguities in handling raw bytes (as string or []byte)  
        during schema-less decoding (decoding into a nil interface{})
      - RPC Server/Client Codec for msgpack-rpc protocol defined at: 
        https://github.com/msgpack-rpc/msgpack-rpc/blob/master/spec.md
  - Fast Paths for some container types:  
    For some container types, we circumvent reflection and its associated overhead
    and allocation costs, and encode/decode directly. These types are:  
	    []interface{}
	    []int
	    []string
	    map[interface{}]interface{}
	    map[int]interface{}
	    map[string]interface{}

## Extension Support

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

## RPC

RPC Client and Server Codecs are implemented, so the codecs can be used
with the standard net/rpc package.

## Usage

Typical usage model:

    // create and configure Handle
    var (
      bh codec.BincHandle
      mh codec.MsgpackHandle
    )

    mh.MapType = reflect.TypeOf(map[string]interface{}(nil))
    
    // configure extensions
    // e.g. for msgpack, define functions and enable Time support for tag 1
    // mh.AddExt(reflect.TypeOf(time.Time{}), 1, myMsgpackTimeEncodeExtFn, myMsgpackTimeDecodeExtFn)

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

## Representative Benchmark Results

A sample run of benchmark using "go test -bi -bench=. -benchmem":

    /proc/cpuinfo: Intel(R) Core(TM) i7-2630QM CPU @ 2.00GHz (HT)
    
    ..............................................
    BENCHMARK INIT: 2013-10-16 11:02:50.345970786 -0400 EDT
    To run full benchmark comparing encodings (MsgPack, Binc, JSON, GOB, etc), use: "go test -bench=."
    Benchmark: 
    	Struct recursive Depth:             1
    	ApproxDeepSize Of benchmark Struct: 4694 bytes
    Benchmark One-Pass Run:
    	 v-msgpack: len: 1600 bytes
    	      bson: len: 3025 bytes
    	   msgpack: len: 1560 bytes
    	      binc: len: 1187 bytes
    	       gob: len: 1972 bytes
    	      json: len: 2538 bytes
    ..............................................
    PASS
    Benchmark__Msgpack____Encode	   50000	     54359 ns/op	   14953 B/op	      83 allocs/op
    Benchmark__Msgpack____Decode	   10000	    106531 ns/op	   14990 B/op	     410 allocs/op
    Benchmark__Binc_NoSym_Encode	   50000	     53956 ns/op	   14966 B/op	      83 allocs/op
    Benchmark__Binc_NoSym_Decode	   10000	    103751 ns/op	   14529 B/op	     386 allocs/op
    Benchmark__Binc_Sym___Encode	   50000	     65961 ns/op	   17130 B/op	      88 allocs/op
    Benchmark__Binc_Sym___Decode	   10000	    106310 ns/op	   15857 B/op	     287 allocs/op
    Benchmark__Gob________Encode	   10000	    135944 ns/op	   21189 B/op	     237 allocs/op
    Benchmark__Gob________Decode	    5000	    405390 ns/op	   83460 B/op	    1841 allocs/op
    Benchmark__Json_______Encode	   20000	     79412 ns/op	   13874 B/op	     102 allocs/op
    Benchmark__Json_______Decode	   10000	    247979 ns/op	   14202 B/op	     493 allocs/op
    Benchmark__Bson_______Encode	   10000	    121762 ns/op	   27814 B/op	     514 allocs/op
    Benchmark__Bson_______Decode	   10000	    162126 ns/op	   16514 B/op	     789 allocs/op
    Benchmark__VMsgpack___Encode	   50000	     69155 ns/op	   12370 B/op	     344 allocs/op
    Benchmark__VMsgpack___Decode	   10000	    151609 ns/op	   20307 B/op	     571 allocs/op
    ok  	ugorji.net/codec	30.827s

To run full benchmark suite (including against vmsgpack and bson), 
see notes in ext\_dep\_test.go

