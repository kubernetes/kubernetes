/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vix

import (
	"encoding/base64"
	"encoding/binary"
	"math"
	"reflect"
	"testing"
)

func TestToolsStateProperties(t *testing.T) {
	// captured from vmtoolsd
	str := `lxEAAAIAAAAoAAAATGludXggNC40LjAtMjEtZ2VuZXJpYyBVYnVudHUgMTYuMDQgTFRTAKgRAAACAAAACgAAAHVidW50dS02NACfEQAAAgAAAA0AAABWTXdhcmUgVG9vbHMAlBEAAAIAAAAVAAAAMTAuMC41IGJ1aWxkLTMyMjc4NzIAmREAAAIAAAATAAAAdWJ1bnR1LTE2MDQtdm13YXJlAJURAAABAAAABAAAAAEAAACWEQAAAQAAAAQAAAABAAAAmBEAAAIAAAABAAAAAMsAAAACAAAAEQAAAC90bXAvdm13YXJlLXJvb3QApxEAAAEAAAAEAAAAQAAAAK0RAAACAAAACgAAAC9tbnQvaGdmcwC8EQAAAwAAAAEAAAAAvREAAAMAAAABAAAAAL4RAAADAAAAAQAAAAC/EQAAAwAAAAEAAAAAwBEAAAMAAAABAAAAAMERAAADAAAAAQAAAADCEQAAAwAAAAEAAAAAwxEAAAMAAAABAAAAAMQRAAADAAAAAQAAAADFEQAAAwAAAAEAAAAAxhEAAAMAAAABAAAAAMcRAAADAAAAAQAAAADIEQAAAwAAAAEAAAAAyREAAAMAAAABAAAAAMoRAAADAAAAAQAAAADLEQAAAwAAAAEAAAAAzBEAAAMAAAABAAAAAM0RAAADAAAAAQAAAADOEQAAAwAAAAEAAAAAzxEAAAMAAAABAAAAANARAAADAAAAAQAAAADREQAAAwAAAAEAAAAA0hEAAAMAAAABAAAAANMRAAADAAAAAQAAAADUEQAAAwAAAAEAAAAA1REAAAMAAAABAAAAANYRAAADAAAAAQAAAADXEQAAAwAAAAEAAAAA2BEAAAMAAAABAAAAAA==`

	data, err := base64.StdEncoding.DecodeString(str)
	if err != nil {
		t.Fatal(err)
	}

	var props PropertyList

	err = props.UnmarshalBinary(data)
	if err != nil {
		t.Error(err)
	}

	data2, err := props.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	str2 := base64.StdEncoding.EncodeToString(data2)
	if str != str2 {
		t.Error("encoding mismatch")
	}
}

func TestMarshalProperties(t *testing.T) {
	in := PropertyList{
		NewInt32Property(1, math.MaxInt32),
		NewStringProperty(2, "foo"),
		NewBoolProperty(3, true),
		NewInt64Property(4, math.MaxInt64),
		NewBlobProperty(5, []byte("deadbeef")),
	}

	buf, err := in.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	var out PropertyList

	err = out.UnmarshalBinary(buf)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(in, out) {
		t.Error("marshal mismatch")
	}

}

// hit unmarshal error paths
func TestUnmarshalPropertiesErrors(t *testing.T) {
	props := PropertyList{
		NewBoolProperty(1, true),
		NewStringProperty(1, "foo"),
		NewBlobProperty(2, []byte("deadbeef")),
	}

	props[0].header.Kind = 0xff

	for _, prop := range props {
		buf, _ := prop.MarshalBinary()

		for i, l := range []int{1, binary.Size(prop.header)} {
			err := prop.UnmarshalBinary(buf[:l])
			if err == nil {
				t.Errorf("test %d (len=%d) expected error", i, l)
			}
		}
	}

	buf, _ := props.MarshalBinary()
	err := props.UnmarshalBinary(buf)
	if err == nil {
		t.Error("expected error")
	}
}
