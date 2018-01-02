# Custom types

Custom types is a gogo protobuf extensions that allows for using a custom
struct type to decorate the underlying structure of the protocol message.

# How to use

## Defining the protobuf message

```proto
message CustomType {
  optional ProtoType Field = 1 [(gogoproto.customtype) = "T"];
}

message ProtoType {
  optional string Field = 1;
}
```

or alternatively you can declare the field type in the protocol message to be
`bytes`:

```proto
message BytesCustomType {
  optional bytes Field = 1 [(gogoproto.customtype) = "T"];
}
```

The downside of using `bytes` is that it makes it harder to generate protobuf
code in other languages. In either case, it is the user responsibility to
ensure that the custom type marshals and unmarshals to the expected wire
format. That is, in the first example, gogo protobuf will not attempt to ensure
that the wire format of `ProtoType` and `T` are wire compatible.

## Custom type method signatures

The custom type must define the following methods with the given
signatures. Assuming the custom type is called `T`:

```go
func (t T) Marshal() ([]byte, error) {}
func (t *T) MarshalTo(data []byte) (n int, err error) {}
func (t *T) Unmarshal(data []byte) error {}

func (t T) MarshalJSON() ([]byte, error) {}
func (t *T) UnmarshalJSON(data []byte) error {}

// only required if the compare option is set
func (t T) Compare(other T) int {}
// only required if the equal option is set
func (t T) Equal(other T) bool {}
// only required if populate option is set
func NewPopulatedT(r randyThetest) *T {}
```

Check [t.go](test/t.go) for a full example

# Warnings and issues

`Warning about customtype: It is your responsibility to test all cases of your marshaling, unmarshaling and size methods implemented for your custom type.`

Issues with customtype include:
  * <a href="https://github.com/gogo/protobuf/issues/199">A Bytes method is not allowed.<a/>
  * <a href="https://github.com/gogo/protobuf/issues/132">Defining a customtype as a fake proto message is broken.</a>
  * <a href="https://github.com/gogo/protobuf/issues/147">proto.Clone is broken.</a>
  * <a href="https://github.com/gogo/protobuf/issues/125">Using a proto message as a customtype is not allowed.</a>
  * <a href="https://github.com/gogo/protobuf/issues/200">cusomtype of type map can not UnmarshalText</a>
  * <a href="https://github.com/gogo/protobuf/issues/201">customtype of type struct cannot jsonpb unmarshal</a>
