// Package aws provides the core SDK's utilities and shared types. Use this package's
// utilities to simplify setting and reading API operations parameters.
//
// Value and Pointer Conversion Utilities
//
// This package includes a helper conversion utility for each scalar type the SDK's
// API use. These utilities make getting a pointer of the scalar, and dereferencing
// a pointer easier.
//
// Each conversion utility comes in two forms. Value to Pointer and Pointer to Value.
// The Pointer to value will safely dereference the pointer and return its value.
// If the pointer was nil, the scalar's zero value will be returned.
//
// The value to pointer functions will be named after the scalar type. So get a
// *string from a string value use the "String" function. This makes it easy to
// to get pointer of a literal string value, because getting the address of a
// literal requires assigning the value to a variable first.
//
//    var strPtr *string
//
//    // Without the SDK's conversion functions
//    str := "my string"
//    strPtr = &str
//
//    // With the SDK's conversion functions
//    strPtr = aws.String("my string")
//
//    // Convert *string to string value
//    str = aws.StringValue(strPtr)
//
// In addition to scalars the aws package also includes conversion utilities for
// map and slice for commonly types used in API parameters. The map and slice
// conversion functions use similar naming pattern as the scalar conversion
// functions.
//
//    var strPtrs []*string
//    var strs []string = []string{"Go", "Gophers", "Go"}
//
//    // Convert []string to []*string
//    strPtrs = aws.StringSlice(strs)
//
//    // Convert []*string to []string
//    strs = aws.StringValueSlice(strPtrs)
//
// SDK Default HTTP Client
//
// The SDK will use the http.DefaultClient if a HTTP client is not provided to
// the SDK's Session, or service client constructor. This means that if the
// http.DefaultClient is modified by other components of your application the
// modifications will be picked up by the SDK as well.
//
// In some cases this might be intended, but it is a better practice to create
// a custom HTTP Client to share explicitly through your application. You can
// configure the SDK to use the custom HTTP Client by setting the HTTPClient
// value of the SDK's Config type when creating a Session or service client.
package aws
