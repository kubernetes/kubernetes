# gogoprotobuf Extensions

Here is an [example.proto](https://github.com/gogo/protobuf/blob/master/test/example/example.proto) which uses most of the gogoprotobuf code generation plugins.

Please also look at the example [Makefile](https://github.com/gogo/protobuf/blob/master/test/example/Makefile) which shows how to specify the `descriptor.proto` and `gogo.proto` in your proto_path

The documentation at [http://godoc.org/github.com/gogo/protobuf/gogoproto](http://godoc.org/github.com/gogo/protobuf/gogoproto) describes the extensions made to goprotobuf in more detail.

Also see [http://godoc.org/github.com/gogo/protobuf/plugin/](http://godoc.org/github.com/gogo/protobuf/plugin/) for documentation of each of the extensions which have their own plugins.

# Fast Marshalling and Unmarshalling

Generating a `Marshal`, `MarshalTo`, `Size` (or `ProtoSize`) and `Unmarshal` method for a struct results in faster marshalling and unmarshalling than when using reflect.

See [BenchComparison](https://github.com/gogo/protobuf/blob/master/bench.md) for a comparison between reflect and generated code used for marshalling and unmarshalling.

<table>
<tr><td><b>Name</b></td><td><b>Option</b></td><td><b>Type</b></td><td><b>Description</b></td><td><b>Default</b></td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/marshalto">marshaler</a></td><td>Message</td><td>bool</td><td>if true, a Marshal and MarshalTo method is generated for the specific message</td><td>false</td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/size">sizer</a></td><td>Message</td><td>bool</td><td>if true, a Size method is generated for the specific message</td><td>false</td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/unmarshal">unmarshaler</a></td><td> Message </td><td> bool </td><td> if true, an Unmarshal method is generated for the specific message </td><td> false</td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/size">protosizer</a></td><td>Message</td><td>bool</td><td>if true, a ProtoSize method is generated for the specific message</td><td>false</td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/marshalto"> unsafe_marshaler</a> (deprecated) </td><td> Message </td><td> bool </td><td> if true, a Marshal and MarshalTo method is generated. </td><td> false</td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/unmarshal">unsafe_unmarshaler</a> (deprecated) </td><td> Message </td><td> bool </td><td> if true, an Unmarshal method is generated. </td><td> false</td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/marshalto">stable_marshaler</a></td><td> Message </td><td> bool </td><td> if true, a Marshal and MarshalTo method is generated for the specific message, but unlike marshaler the output is guaranteed to be deterministic, at the sacrifice of some speed</td><td> false </td></tr>
<tr><td>typedecl (beta)</td><td> Message </td><td> bool </td><td> if false, type declaration of the message is excluded from the generated output. Requires the marshaler and unmarshaler to be generated.</td><td> true </td></tr>
</table>

# More Canonical Go Structures

Lots of times working with a goprotobuf struct will lead you to a place where you create another struct that is easier to work with and then have a function to copy the values between the two structs.

You might also find that basic structs that started their life as part of an API need to be sent over the wire. With gob, you could just send it. With goprotobuf, you need to make a new struct.

`gogoprotobuf` tries to fix these problems with the nullable, embed, customtype, customname, casttype, castkey and castvalue field extensions.

<table>
<tr><td><b>Name</b></td><td><b>Option</b></td><td><b>Type</b></td><td><b>Description</b></td><td><b>Default</b></td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/gogoproto">nullable</a></td><td> Field </td><td> bool </td><td> if false, a field is generated without a pointer (see warning below). </td><td> true </td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/gogoproto">embed</a></td><td> Field </td><td> bool </td><td> if true, the field is generated as an embedded field. </td><td> false </td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/gogoproto">customtype</a> </td><td> Field </td><td> string </td><td> It works with the Marshal and Unmarshal methods, to allow you to have your own types in your struct, but marshal to bytes. For example, custom.Uuid or custom.Fixed128. For more information please refer to the <a href="custom_types.md">CustomTypes</a> document </td><td> goprotobuf type </td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/gogoproto"> customname</a> (beta) </td><td> Field </td><td> string </td><td> Changes the generated fieldname. This is especially useful when generated methods conflict with fieldnames. </td><td> goprotobuf field name </td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/gogoproto"> casttype</a> (beta) </td><td> Field </td><td> string </td><td> Changes the generated field type. It assumes that this type is castable to the original goprotobuf field type.  It currently does not support maps, structs or enums. </td><td> goprotobuf field type </td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/gogoproto"> castkey </a> (beta) </td><td> Field </td><td> string </td><td> Changes the generated fieldtype for a map key.  All generated code assumes that this type is castable to the protocol buffer field type.  Only supported on maps. </td><td> goprotobuf field type </td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/gogoproto"> castvalue </a> (beta) </td><td> Field </td><td> string </td><td> Changes the generated fieldtype for a map value.  All generated code assumes that this type is castable to the protocol buffer field type.  Only supported on maps. </td><td> goprotobuf field type </td></tr>
<tr><td>enum_customname  (beta)</td><td> Enum </td><td> string </td><td>Sets the type name of an enum. If goproto_enum_prefix is enabled, this value will be used as a prefix when generating enum values.</td><td>goprotobuf enum type name. Helps with golint issues.</td></tr>
<tr><td>enumdecl (beta)</td><td> Enum </td><td> bool </td><td> if false, type declaration of the enum is excluded from the generated output. Requires the marshaler and unmarshaler to be generated. </td><td> true </td></tr>
<tr><td>enumvalue_customname (beta) </td><td> Enum Value </td><td> string </td><td>Changes the generated enum name.  Helps with golint issues.</td><td>goprotobuf enum value name</td></tr>
<tr><td><a href="https://github.com/gogo/protobuf/blob/master/test/types/types.proto">stdtime</a></td><td> Timestamp Field </td><td> bool </td><td>Changes the Well Known Timestamp Type to time.Time</td><td>Timestamp</td></tr>
<tr><td><a href="https://github.com/gogo/protobuf/blob/master/test/types/types.proto">stdduration</a></td><td> Duration Field </td><td> bool </td><td>Changes the Well Known Duration Type to time.Duration</td><td>Duration</td></tr>
</table>

`Warning about nullable: according to the Protocol Buffer specification, you should be able to tell whether a field is set or unset. With the option nullable=false this feature is lost, since your non-nullable fields will always be set.`

# Goprotobuf Compatibility

Gogoprotobuf is compatible with Goprotobuf, because it is compatible with protocol buffers (see the section on tests below).

Gogoprotobuf generates the same code as goprotobuf if no extensions are used.

The enumprefix, getters and stringer extensions can be used to remove some of the unnecessary code generated by goprotobuf.

<table>
<tr><td><b>Name</b></td><td><b>Option</b></td><td><b>Type</b></td><td><b>Description</b></td><td><b>Default</b></td></tr>
<tr><td> gogoproto_import </td><td> File </td><td> bool </td><td> if false, the generated code imports github.com/golang/protobuf/proto instead of github.com/gogo/protobuf/proto. </td><td> true </td></tr>
<tr><td> goproto_enum_prefix </td><td> Enum </td><td> bool </td><td> if false, generates the enum constant names without the messagetype prefix </td><td> true </td></tr>
<tr><td> goproto_getters </td><td> Message </td><td> bool </td><td> if false, the message is generated without get methods, this is useful when you would rather want to use face </td><td> true </td></tr>
<tr><td> goproto_stringer </td><td> Message </td><td> bool </td><td> if false, the message is generated without the default string method, this is useful for rather using stringer </td><td> true </td></tr>
<tr><td> goproto_enum_stringer (experimental) </td><td> Enum </td><td> bool </td><td> if false, the enum is generated without the default string method, this is useful for rather using enum_stringer </td><td> true </td></tr>
<tr><td> goproto_extensions_map (beta) </td><td> Message </td><td> bool </td><td> if false, the extensions field is generated as type []byte instead of type map[int32]proto.Extension </td><td> true </td></tr>
<tr><td> goproto_unrecognized (beta) </td><td> Message </td><td> bool </td><td>if false, XXX_unrecognized field is not generated. This is useful to reduce GC pressure at the cost of losing information about unrecognized fields. </td><td> true </td></tr>
<tr><td> goproto_unkeyed (alpha) </td><td> Message </td><td> bool </td><td>if false, XXX_unkeyed field is not generated. </td><td> true </td></tr>
<tr><td> goproto_sizecache (alpha) </td><td> Message </td><td> bool </td><td>if false, XXX_sizecache field is not generated. </td><td> true </td></tr>
<tr><td> goproto_registration (beta) </td><td> File </td><td> bool </td><td>if true, the generated files will register all messages and types against both gogo/protobuf and golang/protobuf. This is necessary when using third-party packages which read registrations from golang/protobuf (such as the grpc-gateway). </td><td> false </td></tr>
<tr><td> message_name </td><td> Message </td><td> bool </td><td>if true, a `XXX_MessageName()` method is generated that returns the message's name.  This is useful for grpc-gateway compatibility.</td><td> false </td></tr>
</table>

# Less Typing

The Protocol Buffer language is very parseable and extra code can be easily generated for structures.

Helper methods, functions and interfaces can be generated by triggering certain extensions like gostring.

<table>
<tr><td><b>Name</b></td><td><b>Option</b></td><td><b>Type</b></td><td><b>Description</b></td><td><b>Default</b></td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/gostring">gostring</a></td><td> Message </td><td> bool </td><td> if true, a `GoString` method is generated. This returns a string representing valid go code to reproduce the current state of the struct. </td><td> false </td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/union"> onlyone</a> </td><td> Message </td><td> bool </td><td> if true, all fields must be nullable and only one of the fields may be set, like a union. Two methods are generated: `GetValue() interface{}` and `SetValue(v interface{}) (set bool)`. These provide easier interaction with a union. </td><td> false </td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/equal"> equal</a></td><td> Message </td><td> bool </td><td> if true, an Equal method is generated </td><td> false </td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/compare"> compare</a></td><td> Message </td><td> bool </td><td> if true, a Compare method is generated.  This is very useful for quickly implementing sort on a list of protobuf structs </td><td> false </td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/equal"> verbose_equal</a> </td><td> Message </td><td> bool </td><td> if true, a verbose equal method is generated for the message. This returns an error which describes the exact element which is not equal to the exact element in the other struct. </td><td> false </td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/stringer"> stringer</a> </td><td> Message </td><td> bool </td><td> if true, a String method is generated for the message. </td><td> false </td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/face">face</a> </td><td> Message </td><td> bool </td><td> if true, a function will be generated which can convert a structure which satisfies an interface (face) to the specified structure. This interface contains getters for each of the fields in the struct. The specified struct is also generated with the getters. This allows it to satisfy its own face. </td><td> false </td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/description"> description</a> (beta) </td><td> Message </td><td> bool </td><td> if true, a Description method is generated for the message. </td><td> false </td></tr>
<tr><td> <a href="http://godoc.org/github.com/gogo/protobuf/plugin/populate"> populate</a> </td><td> Message </td><td> bool </td><td> if true, a `NewPopulated<MessageName>` function is generated. This is necessary for  generated tests. </td><td> false </td></tr>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/enumstringer"> enum_stringer</a> (experimental) </td><td> Enum </td><td> bool </td><td> if true, a String method is generated for an Enum </td><td> false </td></tr>
</table>

Issues with Compare include:
  * <a href="https://github.com/gogo/protobuf/issues/221">Oneof is not supported yet</a>
  * <a href="https://github.com/gogo/protobuf/issues/230">Not all Well Known Types are supported yet</a>
  * <a href="https://github.com/gogo/protobuf/issues/231">Maps are not supported</a>

# Peace of Mind

Test and Benchmark generation is done with the following extensions:

<table>
<tr><td><a href="http://godoc.org/github.com/gogo/protobuf/plugin/testgen">testgen</a> </td><td> Message </td><td> bool </td><td> if true, tests are generated for proto, json and prototext marshalling as well as for some of the other enabled plugins </td><td> false </td></tr>
<tr><td> benchgen </td><td> Message </td><td> bool </td><td> if true, benchmarks are generated for proto, json and prototext marshalling as well as for some of the other enabled plugins </td><td> false </td></tr>
</table>

# More Serialization Formats

Other serialization formats like xml and json typically use reflect to marshal and unmarshal structured data.  Manipulating these structs into something other than the default Go requires editing tags.  The following extensions provide ways of editing these tags for the generated protobuf structs.

<table>
<tr><td><a href="https://github.com/gogo/protobuf/blob/master/test/tags/tags.proto">jsontag</a> (beta) </td><td> Field </td><td> string </td><td> if set, the json tag value between the double quotes is replaced with this string </td><td> fieldname </td></tr>
<tr><td><a href="https://github.com/gogo/protobuf/blob/master/test/tags/tags.proto">moretags</a> (beta) </td><td> Field </td><td> string </td><td> if set, this string is appended to the tag string </td><td> empty </td></tr>
</table>

<a href="https://groups.google.com/forum/#!topic/gogoprotobuf/xmFnqAS6MIc">Here is a longer explanation of jsontag and moretags</a>

# File Options

Each of the boolean message and enum extensions also have a file extension:

  * `marshaler_all`
  * `sizer_all`
  * `protosizer_all`
  * `unmarshaler_all`
  * `unsafe_marshaler_all`
  * `unsafe_unmarshaler_all`
  * `stable_marshaler_all`
  * `goproto_enum_prefix_all`
  * `goproto_getters_all`
  * `goproto_stringer_all`
  * `goproto_enum_stringer_all`
  * `goproto_extensions_map_all`
  * `goproto_unrecognized_all`
  * `goproto_unkeyed_all`
  * `goproto_sizecache_all`
  * `gostring_all`
  * `onlyone_all`
  * `equal_all`
  * `compare_all`
  * `verbose_equal_all`
  * `stringer_all`
  * `enum_stringer_all`
  * `face_all`
  * `description_all`
  * `populate_all`
  * `testgen_all`
  * `benchgen_all`
  * `enumdecl_all`
  * `typedecl_all`
  * `messagename_all`

Each of these are the same as their Message Option counterparts, except they apply to all messages in the file.  Their Message option counterparts can also be used to overwrite their effect.

# Tests

  * The normal barrage of tests are run with: `make tests`
  * A few weird tests: `make testall`
  * Tests for compatibility with [golang/protobuf](https://github.com/golang/protobuf) are handled by a different project [harmonytests](https://github.com/gogo/harmonytests), since it requires goprotobuf.
  * Cross version tests are made with [Travis CI](https://travis-ci.org/gogo/protobuf).
  * GRPC Tests are also handled by a different project [grpctests](https://github.com/gogo/grpctests), since it depends on a lot of grpc libraries.
  * Thanks to [go-fuzz](https://github.com/dvyukov/go-fuzz/) we have proper [fuzztests](https://github.com/gogo/fuzztests).

