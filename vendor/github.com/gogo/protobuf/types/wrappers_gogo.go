// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2018, The GoGo Authors. All rights reserved.
// http://github.com/gogo/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package types

func NewPopulatedStdDouble(r randyWrappers, easy bool) *float64 {
	v := NewPopulatedDoubleValue(r, easy)
	return &v.Value
}

func SizeOfStdDouble(v float64) int {
	pv := &DoubleValue{Value: v}
	return pv.Size()
}

func StdDoubleMarshal(v float64) ([]byte, error) {
	size := SizeOfStdDouble(v)
	buf := make([]byte, size)
	_, err := StdDoubleMarshalTo(v, buf)
	return buf, err
}

func StdDoubleMarshalTo(v float64, data []byte) (int, error) {
	pv := &DoubleValue{Value: v}
	return pv.MarshalTo(data)
}

func StdDoubleUnmarshal(v *float64, data []byte) error {
	pv := &DoubleValue{}
	if err := pv.Unmarshal(data); err != nil {
		return err
	}
	*v = pv.Value
	return nil
}
func NewPopulatedStdFloat(r randyWrappers, easy bool) *float32 {
	v := NewPopulatedFloatValue(r, easy)
	return &v.Value
}

func SizeOfStdFloat(v float32) int {
	pv := &FloatValue{Value: v}
	return pv.Size()
}

func StdFloatMarshal(v float32) ([]byte, error) {
	size := SizeOfStdFloat(v)
	buf := make([]byte, size)
	_, err := StdFloatMarshalTo(v, buf)
	return buf, err
}

func StdFloatMarshalTo(v float32, data []byte) (int, error) {
	pv := &FloatValue{Value: v}
	return pv.MarshalTo(data)
}

func StdFloatUnmarshal(v *float32, data []byte) error {
	pv := &FloatValue{}
	if err := pv.Unmarshal(data); err != nil {
		return err
	}
	*v = pv.Value
	return nil
}
func NewPopulatedStdInt64(r randyWrappers, easy bool) *int64 {
	v := NewPopulatedInt64Value(r, easy)
	return &v.Value
}

func SizeOfStdInt64(v int64) int {
	pv := &Int64Value{Value: v}
	return pv.Size()
}

func StdInt64Marshal(v int64) ([]byte, error) {
	size := SizeOfStdInt64(v)
	buf := make([]byte, size)
	_, err := StdInt64MarshalTo(v, buf)
	return buf, err
}

func StdInt64MarshalTo(v int64, data []byte) (int, error) {
	pv := &Int64Value{Value: v}
	return pv.MarshalTo(data)
}

func StdInt64Unmarshal(v *int64, data []byte) error {
	pv := &Int64Value{}
	if err := pv.Unmarshal(data); err != nil {
		return err
	}
	*v = pv.Value
	return nil
}
func NewPopulatedStdUInt64(r randyWrappers, easy bool) *uint64 {
	v := NewPopulatedUInt64Value(r, easy)
	return &v.Value
}

func SizeOfStdUInt64(v uint64) int {
	pv := &UInt64Value{Value: v}
	return pv.Size()
}

func StdUInt64Marshal(v uint64) ([]byte, error) {
	size := SizeOfStdUInt64(v)
	buf := make([]byte, size)
	_, err := StdUInt64MarshalTo(v, buf)
	return buf, err
}

func StdUInt64MarshalTo(v uint64, data []byte) (int, error) {
	pv := &UInt64Value{Value: v}
	return pv.MarshalTo(data)
}

func StdUInt64Unmarshal(v *uint64, data []byte) error {
	pv := &UInt64Value{}
	if err := pv.Unmarshal(data); err != nil {
		return err
	}
	*v = pv.Value
	return nil
}
func NewPopulatedStdInt32(r randyWrappers, easy bool) *int32 {
	v := NewPopulatedInt32Value(r, easy)
	return &v.Value
}

func SizeOfStdInt32(v int32) int {
	pv := &Int32Value{Value: v}
	return pv.Size()
}

func StdInt32Marshal(v int32) ([]byte, error) {
	size := SizeOfStdInt32(v)
	buf := make([]byte, size)
	_, err := StdInt32MarshalTo(v, buf)
	return buf, err
}

func StdInt32MarshalTo(v int32, data []byte) (int, error) {
	pv := &Int32Value{Value: v}
	return pv.MarshalTo(data)
}

func StdInt32Unmarshal(v *int32, data []byte) error {
	pv := &Int32Value{}
	if err := pv.Unmarshal(data); err != nil {
		return err
	}
	*v = pv.Value
	return nil
}
func NewPopulatedStdUInt32(r randyWrappers, easy bool) *uint32 {
	v := NewPopulatedUInt32Value(r, easy)
	return &v.Value
}

func SizeOfStdUInt32(v uint32) int {
	pv := &UInt32Value{Value: v}
	return pv.Size()
}

func StdUInt32Marshal(v uint32) ([]byte, error) {
	size := SizeOfStdUInt32(v)
	buf := make([]byte, size)
	_, err := StdUInt32MarshalTo(v, buf)
	return buf, err
}

func StdUInt32MarshalTo(v uint32, data []byte) (int, error) {
	pv := &UInt32Value{Value: v}
	return pv.MarshalTo(data)
}

func StdUInt32Unmarshal(v *uint32, data []byte) error {
	pv := &UInt32Value{}
	if err := pv.Unmarshal(data); err != nil {
		return err
	}
	*v = pv.Value
	return nil
}
func NewPopulatedStdBool(r randyWrappers, easy bool) *bool {
	v := NewPopulatedBoolValue(r, easy)
	return &v.Value
}

func SizeOfStdBool(v bool) int {
	pv := &BoolValue{Value: v}
	return pv.Size()
}

func StdBoolMarshal(v bool) ([]byte, error) {
	size := SizeOfStdBool(v)
	buf := make([]byte, size)
	_, err := StdBoolMarshalTo(v, buf)
	return buf, err
}

func StdBoolMarshalTo(v bool, data []byte) (int, error) {
	pv := &BoolValue{Value: v}
	return pv.MarshalTo(data)
}

func StdBoolUnmarshal(v *bool, data []byte) error {
	pv := &BoolValue{}
	if err := pv.Unmarshal(data); err != nil {
		return err
	}
	*v = pv.Value
	return nil
}
func NewPopulatedStdString(r randyWrappers, easy bool) *string {
	v := NewPopulatedStringValue(r, easy)
	return &v.Value
}

func SizeOfStdString(v string) int {
	pv := &StringValue{Value: v}
	return pv.Size()
}

func StdStringMarshal(v string) ([]byte, error) {
	size := SizeOfStdString(v)
	buf := make([]byte, size)
	_, err := StdStringMarshalTo(v, buf)
	return buf, err
}

func StdStringMarshalTo(v string, data []byte) (int, error) {
	pv := &StringValue{Value: v}
	return pv.MarshalTo(data)
}

func StdStringUnmarshal(v *string, data []byte) error {
	pv := &StringValue{}
	if err := pv.Unmarshal(data); err != nil {
		return err
	}
	*v = pv.Value
	return nil
}
func NewPopulatedStdBytes(r randyWrappers, easy bool) *[]byte {
	v := NewPopulatedBytesValue(r, easy)
	return &v.Value
}

func SizeOfStdBytes(v []byte) int {
	pv := &BytesValue{Value: v}
	return pv.Size()
}

func StdBytesMarshal(v []byte) ([]byte, error) {
	size := SizeOfStdBytes(v)
	buf := make([]byte, size)
	_, err := StdBytesMarshalTo(v, buf)
	return buf, err
}

func StdBytesMarshalTo(v []byte, data []byte) (int, error) {
	pv := &BytesValue{Value: v}
	return pv.MarshalTo(data)
}

func StdBytesUnmarshal(v *[]byte, data []byte) error {
	pv := &BytesValue{}
	if err := pv.Unmarshal(data); err != nil {
		return err
	}
	*v = pv.Value
	return nil
}
