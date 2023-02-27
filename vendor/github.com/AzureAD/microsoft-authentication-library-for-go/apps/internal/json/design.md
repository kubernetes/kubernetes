# JSON Package Design
Author: John Doak(jdoak@microsoft.com)

## Why?

This project needs a special type of marshal/unmarshal not directly supported
by the encoding/json package. 

The need revolves around a few key wants/needs:
- unmarshal and marshal structs representing JSON messages
- fields in the messgage not in the struct must be maintained when unmarshalled
- those same fields must be marshalled back when encoded again

The initial version used map[string]interface{} to put in the keys that
were known and then any other keys were put into a field called AdditionalFields.

This has a few negatives:
- Dual marshaling/unmarshalling is required
- Adding a struct field requires manually adding a key by name to be encoded/decoded from the map (which is a loosely coupled construct), which can lead to bugs that aren't detected or have bad side effects
- Tests can become quickly disconnected if those keys aren't put
in tests as well. So you think you have support working, but you
don't. Existing tests were found that didn't test the marshalling output.
- There is no enforcement that if AdditionalFields is required on one struct, it should be on all containers
that don't have custom marshal/unmarshal.

This package aims to support our needs by providing custom Marshal()/Unmarshal() functions.

This prevents all the negatives in the initial solution listed above. However, it does add its own negative:
- Custom encoding/decoding via reflection is messy (as can be  seen in encoding/json itself)

Go proverb: Reflection is never clear
Suggested reading: https://blog.golang.org/laws-of-reflection

## Important design decisions

- We don't want to understand all JSON decoding rules
- We don't want to deal with all the quoting, commas, etc on decode
- Need support for json.Marshaler/Unmarshaler, so we can support types like time.Time
- If struct does not implement json.Unmarshaler, it must have AdditionalFields defined
- We only support root level objects that are \*struct or struct

To faciliate these goals, we will utilize the json.Encoder and json.Decoder.
They provide streaming processing (efficient) and return errors on bad JSON.

Support for json.Marshaler/Unmarshaler allows for us to use non-basic types
that must be specially encoded/decoded (like time.Time objects).

We don't support types that can't customer unmarshal or have AdditionalFields
in order to prevent future devs from forgetting that important field and
generating bad return values.

Support for root level objects of \*struct or struct simply acknowledges the
fact that this is designed only for the purposes listed in the Introduction.
Outside that (like encoding a lone number) should be done with the
regular json package (as it will not have additional fields).

We don't support a few things on json supported reference types and structs:
- \*map: no need for pointers to maps
- \*slice: no need for pointers to slices
- any further pointers on struct after \*struct

There should never be a need for this in Go.

## Design

## State Machines

This uses state machine designs that based upon the Rob Pike talk on 
lexers and parsers: https://www.youtube.com/watch?v=HxaD_trXwRE

This is the most common pattern for state machines in Go and
the model to follow closesly when dealing with streaming 
processing of textual data.

Our state machines are based on the type:
```go
type stateFn func() (stateFn, error)
```

The state machine itself is simply a struct that has methods that
satisfy stateFn. 

Our state machines have a few standard calls
- run(): runs the state machine
- start(): always the first stateFn to be called

All state machines have the following logic:
* run() is called
* start() is called and returns the next stateFn or error
* stateFn is called
    - If returned stateFn(next state) is non-nil, call it
    - If error is non-nil, run() returns the error
    - If stateFn == nil and err == nil, run() return err == nil

## Supporting types

Marshalling/Unmarshalling must support(within top level struct):
- struct
- \*struct
- []struct
- []\*struct
- []map[string]structContainer
- [][]structContainer

**Term note:** structContainer == type that has a struct or \*struct inside it

We specifically do not support []interface or map[string]interface
where the interface value would hold some value with a struct in it.

Those will still marshal/unmarshal, but without support for 
AdditionalFields. 

## Marshalling

The marshalling design will be based around a statemachine design. 

The basic logic is as follows:

* If struct has custom marshaller, call it and return
* If struct has field "AdditionalFields", it must be a map[string]interface{}
* If struct does not have "AdditionalFields", give an error
* Get struct tag detailing json names to go names, create mapping
* For each public field name
    - Write field name out
    - If field value is a struct, recursively call our state machine
    - Otherwise, use the json.Encoder to write out the value

## Unmarshalling

The unmarshalling desin is also based around a statemachine design. The 
basic logic is as follows:

* If struct has custom marhaller, call it
* If struct has field "AdditionalFields", it must be a map[string]interface{}
* Get struct tag detailing json names to go names, create mapping
* For each key found
    - If key exists, 
        - If value is basic type, extract value into struct field using Decoder
        - If value is struct type, recursively call statemachine
    - If key doesn't exist, add it to AdditionalFields if it exists using Decoder
