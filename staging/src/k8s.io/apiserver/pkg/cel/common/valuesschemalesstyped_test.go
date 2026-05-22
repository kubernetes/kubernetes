/*
Copyright The Kubernetes Authors.

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

package common

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
)

type customJSONMarshaler struct {
	Val string
}

func (c customJSONMarshaler) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]string{"marshaled": c.Val})
}

type customPointerJSONMarshaler struct {
	Val string
}

func (c *customPointerJSONMarshaler) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]string{"marshaled": c.Val})
}

type myInt int
type myInt32 int32
type myInt64 int64
type myFloat32 float32
type myString string
type myBool bool
type myFloat float64
type myUint uint
type myUint64 uint64

type TestStruct struct {
	FieldA string `json:"fieldA,omitempty"`
	FieldB int    `json:"fieldB"`
	FieldC *int   `json:"fieldC,omitempty"`
}

type OmitTestStruct struct {
	FieldNoOmit *int `json:"fieldNoOmit"`
	FieldOmit   *int `json:"fieldOmit,omitempty"`
}

// The following Equiv structures mirror those in runtime/converter_test.go
// to verify equivalence between SchemalessTypedToVal and ToUnstructured.
type EquivA struct {
	A int    `json:"aa,omitempty"`
	B string `json:"ab,omitempty"`
	C bool   `json:"ac,omitempty"`
}

type EquivB struct {
	A EquivA            `json:"ba"`
	B string            `json:"bb"`
	C map[string]string `json:"bc"`
	D []string          `json:"bd"`
}

type EquivC struct {
	A      []EquivA `json:"ca"`
	EquivB `json:""`
	C      string         `json:"cc"`
	D      *int64         `json:"cd"`
	E      map[string]int `json:"ce"`
	F      []bool         `json:"cf"`
	G      []int          `json:"cg"`
	H      float32        `json:"ch"`
	I      []interface{}  `json:"ci"`
}

type EquivD struct {
	A []interface{} `json:"da"`
}

type EquivE struct {
	A interface{} `json:"ea"`
}

type EquivF struct {
	A string            `json:"fa"`
	B map[string]string `json:"fb"`
	C []EquivA          `json:"fc"`
	D int               `json:"fd"`
	E float32           `json:"fe"`
	F []string          `json:"ff"`
	G []int             `json:"fg"`
	H []bool            `json:"fh"`
	I []float32         `json:"fi"`
	J []byte            `json:"fj"`
}

type EquivG struct {
	CustomValue1   EquivCustomValue      `json:"customValue1"`
	CustomValue2   *EquivCustomValue     `json:"customValue2"`
	CustomPointer1 EquivCustomPointer    `json:"customPointer1"`
	CustomPointer2 *EquivCustomPointer   `json:"customPointer2"`
	RawExtension1  runtime.RawExtension  `json:"rawExtension1"`
	RawExtension2  *runtime.RawExtension `json:"rawExtension2"`
}

type EquivCustomValue struct {
	Data []byte
}

func (c EquivCustomValue) MarshalJSON() ([]byte, error) {
	if len(c.Data) == 0 {
		return []byte("null"), nil
	}
	return c.Data, nil
}

type EquivCustomPointer struct {
	Data []byte
}

func (c *EquivCustomPointer) MarshalJSON() ([]byte, error) {
	if len(c.Data) == 0 {
		return []byte("null"), nil
	}
	return c.Data, nil
}

type EquivInlineTestPrimitive struct {
	NoNameTagPrimitive          int64 `json:""`
	NoNameTagInlinePrimitive    int64 `json:""`
	NoNameTagOmitemptyPrimitive int64 `json:",omitempty"`
}

type EquivInlineTestAnonymous struct {
	EquivNoTag
	EquivNoNameTag          `json:""`
	EquivNameTag            `json:"nameTagEmbedded"`
	EquivNoNameTagInline    `json:""`
	EquivNoNameTagOmitempty `json:",omitempty"`
}

type EquivInlineTestNamed struct {
	NoTag              EquivNoTag
	NoNameTag          EquivNoNameTag          `json:""`
	NameTag            EquivNameTag            `json:"nameTagEmbedded"`
	NoNameTagInline    EquivNoNameTagInline    `json:""`
	NoNameTagOmitempty EquivNoNameTagOmitempty `json:",omitempty"`
}

type EquivNoTag struct {
	Data0 int `json:"data0"`
}

type EquivNameTag struct {
	Data1 int `json:"data1"`
}

type EquivNoNameTag struct {
	Data2 int `json:"data2"`
}

type EquivNoNameTagInline struct {
	Data3 int `json:"data3"`
}

type EquivNoNameTagOmitempty struct {
	Data4 int `json:"data4"`
}

func assertEquivalence(t *testing.T, val interface{}) {
	unstrMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(val)
	if err != nil {
		t.Fatalf("ToUnstructured failed: %v", err)
	}

	expectedCEL := types.DefaultTypeAdapter.NativeToValue(unstrMap)
	gotCEL := SchemalessTypedToVal(val)

	if gotCEL.Equal(expectedCEL) != types.True {
		t.Errorf("Equivalence mismatch!\nExpected: %v (Type: %T, CEL: %v)\nGot:      %v (Type: %T, CEL: %v)",
			unstrMap, expectedCEL, expectedCEL.Type(), gotCEL.Value(), gotCEL, gotCEL.Type())
	}

	// Verify inverse equality (NativeToValue.Equal(SchemalessTypedToVal)).
	if expectedCEL.Equal(gotCEL) != types.True {
		t.Errorf("Inverse equivalence mismatch!")
	}
}

func TestSchemalessTypedToVal_Equivalence(t *testing.T) {
	timeVal := time.Date(2026, 5, 18, 12, 0, 0, 0, time.UTC)
	microTimeVal := time.Date(2026, 5, 18, 12, 0, 0, 1000, time.UTC)
	cd := int64(12345)

	tests := []struct {
		name string
		val  interface{}
	}{
		{
			name: "corev1.Pod",
			val: &corev1.Pod{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "v1",
					Kind:       "Pod",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
					Labels: map[string]string{
						"app": "test",
					},
					CreationTimestamp: metav1.Time{Time: timeVal},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "test-container",
							Image: "nginx",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU: resource.MustParse("100m"),
								},
							},
						},
					},
				},
			},
		},
		{
			name: "StructA full",
			val: &EquivA{
				A: 12,
				B: "hello",
				C: true,
			},
		},
		{
			name: "StructA empty",
			val:  &EquivA{},
		},
		{
			name: "StructB",
			val: &EquivB{
				A: EquivA{A: 1, B: "a", C: true},
				B: "str",
				C: map[string]string{"k": "v"},
				D: []string{"a", "b"},
			},
		},
		{
			name: "StructC",
			val: &EquivC{
				A: []EquivA{
					{A: 1, B: "a"},
					{A: 2, B: "b"},
				},
				EquivB: EquivB{
					A: EquivA{A: 3, B: "c"},
					B: "embedB",
				},
				C: "helloC",
				D: &cd,
				E: map[string]int{"a": 10, "b": 20},
				F: []bool{true, false},
				G: []int{1, 2, 3},
				H: 3.14,
				I: []interface{}{"any", int64(42), float64(1.23)},
			},
		},
		{
			name: "StructD",
			val: &EquivD{
				A: []interface{}{
					map[string]interface{}{"a": "b"},
					[]interface{}{"c", int64(1)},
				},
			},
		},
		{
			name: "StructE",
			val: &EquivE{
				A: map[string]interface{}{"nested": "val"},
			},
		},
		{
			name: "StructF",
			val: &EquivF{
				A: "fa",
				B: map[string]string{"fb": "val"},
				C: []EquivA{{A: 1}},
				D: 42,
				E: 3.14,
				F: []string{"a"},
				G: []int{1},
				H: []bool{true},
				I: []float32{1.1},
				J: []byte("hello"),
			},
		},
		{
			name: "StructG CustomMarshaling",
			val: &EquivG{
				CustomValue1:   EquivCustomValue{Data: []byte(`{"a":1}`)},
				CustomValue2:   &EquivCustomValue{Data: []byte(`[1,2]`)},
				CustomPointer1: EquivCustomPointer{Data: []byte(`"string"`)},
				CustomPointer2: &EquivCustomPointer{Data: []byte(`42`)},
				RawExtension1:  runtime.RawExtension{Raw: []byte(`{"nestedKey":"nestedVal"}`)},
				RawExtension2:  &runtime.RawExtension{Raw: []byte(`[1,2,3]`)},
			},
		},
		{
			name: "InlinePrimitive empty",
			val:  &EquivInlineTestPrimitive{},
		},
		{
			name: "InlinePrimitive full",
			val: &EquivInlineTestPrimitive{
				NoNameTagPrimitive:          1,
				NoNameTagInlinePrimitive:    2,
				NoNameTagOmitemptyPrimitive: 3,
			},
		},
		{
			name: "InlineAnonymous empty",
			val:  &EquivInlineTestAnonymous{},
		},
		{
			name: "InlineAnonymous full",
			val: &EquivInlineTestAnonymous{
				EquivNoTag:              EquivNoTag{Data0: 100},
				EquivNoNameTag:          EquivNoNameTag{Data2: 200},
				EquivNameTag:            EquivNameTag{Data1: 300},
				EquivNoNameTagInline:    EquivNoNameTagInline{Data3: 400},
				EquivNoNameTagOmitempty: EquivNoNameTagOmitempty{Data4: 500},
			},
		},
		{
			name: "InlineNamed empty",
			val:  &EquivInlineTestNamed{},
		},
		{
			name: "InlineNamed full",
			val: &EquivInlineTestNamed{
				NoTag:              EquivNoTag{Data0: 10},
				NoNameTag:          EquivNoNameTag{Data2: 20},
				NameTag:            EquivNameTag{Data1: 30},
				NoNameTagInline:    EquivNoNameTagInline{Data3: 40},
				NoNameTagOmitempty: EquivNoNameTagOmitempty{Data4: 50},
			},
		},
		{
			name: "OmitAndNil",
			val: &OmitTestStruct{
				FieldNoOmit: nil,
				FieldOmit:   nil,
			},
		},
		{
			name: "K8sTypes",
			val: &struct {
				Time             metav1.Time
				TimePtr          *metav1.Time
				MicroTime        metav1.MicroTime
				Duration         metav1.Duration
				IntOrStringInt   intstr.IntOrString
				IntOrStringStr   intstr.IntOrString
				Quantity         resource.Quantity
				CustomJSON       customJSONMarshaler
				CustomPtrJSON    *customPointerJSONMarshaler
				CustomPtrJSONVal customPointerJSONMarshaler
			}{
				Time:             metav1.NewTime(timeVal),
				MicroTime:        metav1.NewMicroTime(microTimeVal),
				Duration:         metav1.Duration{Duration: 5 * time.Minute},
				IntOrStringInt:   intstr.FromInt32(100),
				IntOrStringStr:   intstr.FromString("http"),
				Quantity:         resource.MustParse("200m"),
				CustomJSON:       customJSONMarshaler{Val: "foo"},
				CustomPtrJSON:    &customPointerJSONMarshaler{Val: "baz"},
				CustomPtrJSONVal: customPointerJSONMarshaler{Val: "qux"},
			},
		},
		{
			name: "Primitives in map",
			val: &map[string]interface{}{
				"nil":        nil,
				"nil_ptr":    (*int)(nil),
				"int_ptr":    func() *int { i := 42; return &i }(),
				"bool_true":  true,
				"bool_false": false,
				"int":        int(12),
				"int32":      int32(32),
				"int64":      int64(64),
				"float32":    float32(3.14),
				"float64":    float64(3.14159),
				"string":     "hello",
				"bytes_nil":  ([]byte)(nil),
				"bytes":      []byte("hello"),
				"myInt":      myInt(42),
				"myInt32":    myInt32(32),
				"myInt64":    myInt64(64),
				"myFloat32":  myFloat32(3.14),
				"myString":   myString("abc"),
				"myBool":     myBool(true),
				"myFloat":    myFloat(1.23),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assertEquivalence(t, tc.val)
		})
	}
}

func TestSchemalessTypedToVal_UnsupportedTypes(t *testing.T) {
	val := make(chan int)
	expectedErr := "unsupported Go type for CEL: chan int"

	got := SchemalessTypedToVal(val)
	if !types.IsError(got) {
		t.Fatalf("expected error, got type: %v, value: %v", got.Type(), got.Value())
	}
	if got.Value().(error).Error() != expectedErr {
		t.Fatalf("got error %q, expected %q", got.Value().(error).Error(), expectedErr)
	}
}

// Note: The equivalence test (assertEquivalence) is not used here because
// runtime.DefaultUnstructuredConverter.ToUnstructured requires structured types (objects/maps)
// and will panic if passed primitive values.
func TestSchemalessTypedToVal_Primitives(t *testing.T) {
	tests := []struct {
		name        string
		val         interface{}
		expected    ref.Val
		expectedErr string
	}{
		{"nil", nil, types.NullValue, ""},
		{"bool_true", true, types.Bool(true), ""},
		{"bool_false", false, types.Bool(false), ""},
		{"int", int(42), types.Int(42), ""},
		{"int32", int32(42), types.Int(42), ""},
		{"int64", int64(42), types.Int(42), ""},
		{"float32", float32(3.14), types.Double(float32(3.14)), ""},
		{"float64", float64(3.14159), types.Double(3.14159), ""},
		{"string", "hello", types.String("hello"), ""},
		{"bytes", []byte("hello"), types.String("aGVsbG8="), ""},
		{"myInt", myInt(42), types.Int(42), ""},
		{"myInt32", myInt32(32), types.Int(32), ""},
		{"uint", uint(42), types.Int(42), ""},
		{"uint8", uint8(255), types.Int(255), ""},
		{"uint16", uint16(65535), types.Int(65535), ""},
		{"uint32", uint32(4294967295), types.Int(4294967295), ""},
		{"uint64 MaxInt64", uint64(9223372036854775807), types.Int(9223372036854775807), ""},
		{"uint64 overflow", uint64(9223372036854775808), nil, "unsigned value 9223372036854775808 does not fit into int64 (overflow)"},
		{"uint64 MaxUint64 overflow", uint64(18446744073709551615), nil, "unsigned value 18446744073709551615 does not fit into int64 (overflow)"},
		{"myUint", myUint(42), types.Int(42), ""},
		{"myUint64 MaxInt64", myUint64(9223372036854775807), types.Int(9223372036854775807), ""},
		{"myUint64 overflow", myUint64(9223372036854775808), nil, "unsigned value 9223372036854775808 does not fit into int64 (overflow)"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := SchemalessTypedToVal(tc.val)
			if types.IsError(got) {
				if tc.expectedErr == "" {
					t.Fatalf("unexpected error: %v", got.Value())
				}
				if got.Value().(error).Error() != tc.expectedErr {
					t.Errorf("expected error %q, got %q", tc.expectedErr, got.Value().(error).Error())
				}
				return
			}
			if tc.expectedErr != "" {
				t.Fatalf("expected error %q, got type: %v, value: %v", tc.expectedErr, got.Type(), got.Value())
			}
			if got.Equal(tc.expected) != types.True {
				t.Errorf("expected %v, got %v", tc.expected.Value(), got.Value())
			}
		})
	}
}

func makePod() *corev1.Pod {
	pod := &corev1.Pod{
		Spec: corev1.PodSpec{
			HostNetwork: false,
			SecurityContext: &corev1.PodSecurityContext{
				RunAsNonRoot: ptr.To(true),
				SELinuxOptions: &corev1.SELinuxOptions{
					Level: "s0:c123,c456",
				},
			},
			Tolerations: []corev1.Toleration{
				{Key: "t1", Operator: corev1.TolerationOpEqual, Value: "v1"},
				{Key: "t2", Operator: corev1.TolerationOpEqual, Value: "v2"},
				{Key: "t3", Operator: corev1.TolerationOpEqual, Value: "v3"},
			},
			Volumes: []corev1.Volume{
				{Name: "v1", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
				{Name: "v2", VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{}}},
				{Name: "v3", VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{}}},
				{Name: "v4", VolumeSource: corev1.VolumeSource{Projected: &corev1.ProjectedVolumeSource{}}},
			},
		},
	}

	for i := 0; i < 5; i++ {
		pod.Spec.Containers = append(pod.Spec.Containers, corev1.Container{
			Name:  fmt.Sprintf("c%d", i),
			Image: "busybox@sha256:1234567890abcdef",
			SecurityContext: &corev1.SecurityContext{
				Capabilities: &corev1.Capabilities{
					Drop: []corev1.Capability{"ALL"},
				},
				RunAsNonRoot: ptr.To(true),
				Privileged:   ptr.To(false),
				RunAsUser:    ptr.To(int64(1000)),
				SeccompProfile: &corev1.SeccompProfile{
					Type: corev1.SeccompProfileTypeRuntimeDefault,
				},
			},
		})
	}
	return pod
}

func getNestedField(val ref.Val) ref.Val {
	key1 := types.String("spec")
	key2 := types.String("securityContext")
	key3 := types.String("seLinuxOptions")
	key4 := types.String("level")
	m1 := val.(traits.Mapper)
	v1 := m1.Get(key1)
	m2 := v1.(traits.Mapper)
	v2 := m2.Get(key2)
	m3 := v2.(traits.Mapper)
	v3 := m3.Get(key3)
	m4 := v3.(traits.Mapper)
	return m4.Get(key4)
}

func BenchmarkAccessNestedField_Unstructured(b *testing.B) {
	obj := makePod()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		unstrMap, _ := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
		_ = getNestedField(types.DefaultTypeAdapter.NativeToValue(unstrMap))
	}
}

func BenchmarkAccessNestedField_SchemalessTyped(b *testing.B) {
	obj := makePod()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		val := SchemalessTypedToVal(obj)
		_ = getNestedField(val)
	}
}

func TestCelTraits(t *testing.T) {
	type dummy struct{ A string }

	tests := []struct {
		name                 string
		val                  interface{}
		expectedType         *types.Type
		validNativeType      reflect.Type
		invalidNativeType    reflect.Type
		invalidTypeToConvert *types.Type
		runTypeSpecific      func(t *testing.T, got ref.Val)
	}{
		{
			name:                 "List",
			val:                  []int{1, 2, 3},
			expectedType:         types.ListType,
			validNativeType:      reflect.TypeFor[[]int](),
			invalidNativeType:    reflect.TypeFor[map[string]int](),
			invalidTypeToConvert: types.MapType,
			runTypeSpecific: func(t *testing.T, got ref.Val) {
				lister, ok := got.(traits.Lister)
				if !ok {
					t.Fatalf("expected traits.Lister")
				}
				if !types.IsError(lister.Get(types.Int(10))) {
					t.Error("expected out of bounds error")
				}
				if !types.IsError(lister.Get(types.String("invalid"))) {
					t.Error("expected invalid key error")
				}
				if !types.IsError(got.(traits.Adder).Add(types.Int(4))) {
					t.Error("expected add non-list error")
				}
				if got.(traits.Container).Contains(types.Int(1)) != types.True {
					t.Error("expected list to contain 1")
				}
				if got.(traits.Container).Contains(types.Int(99)) != types.False {
					t.Error("expected list to not contain 99")
				}

				it := lister.Iterator()
				for it.HasNext() == types.True {
					it.Next()
				}
				if !types.IsError(it.Next()) {
					t.Error("expected exhausted error")
				}
			},
		},
		{
			name:                 "Map",
			val:                  map[string]interface{}{"a": 1},
			expectedType:         types.MapType,
			validNativeType:      reflect.TypeFor[map[string]interface{}](),
			invalidNativeType:    reflect.TypeFor[[]int](),
			invalidTypeToConvert: types.ListType,
			runTypeSpecific: func(t *testing.T, got ref.Val) {
				mapper, ok := got.(traits.Mapper)
				if !ok {
					t.Fatalf("expected traits.Mapper")
				}
				if !types.IsError(mapper.Contains(types.Int(1))) {
					t.Error("expected invalid key error")
				}
				if mapper.Contains(types.String("b")) != types.False {
					t.Error("expected false")
				}
				if v, _ := mapper.Find(types.Int(1)); !types.IsError(v) {
					t.Error("expected error for invalid key")
				}
				if !types.IsError(mapper.Get(types.String("b"))) {
					t.Error("expected not found error")
				}

				it := mapper.Iterator()
				for it.HasNext() == types.True {
					it.Next()
				}
				if !types.IsError(it.Next()) {
					t.Error("expected exhausted error")
				}
			},
		},
		{
			name:                 "Struct",
			val:                  dummy{A: "foo"},
			expectedType:         types.MapType,
			validNativeType:      reflect.TypeFor[dummy](),
			invalidNativeType:    reflect.TypeFor[[]int](),
			invalidTypeToConvert: types.ListType,
			runTypeSpecific: func(t *testing.T, got ref.Val) {
				mapper, ok := got.(traits.Mapper)
				if !ok {
					t.Fatalf("expected traits.Mapper")
				}
				tester, ok := got.(traits.FieldTester)
				if !ok {
					t.Fatalf("expected traits.FieldTester")
				}

				if !types.IsError(mapper.Contains(types.Int(1))) {
					t.Error("expected invalid key error")
				}
				if !types.IsError(tester.IsSet(types.Int(1))) {
					t.Error("expected invalid key error")
				}
				if mapper.Contains(types.String("B")) != types.False {
					t.Error("expected false")
				}
				if tester.IsSet(types.String("B")) != types.False {
					t.Error("expected false")
				}
				if v, _ := mapper.Find(types.Int(1)); !types.IsError(v) {
					t.Error("expected error not found")
				}
				if !types.IsError(mapper.Get(types.String("B"))) {
					t.Error("expected not found error")
				}

				it := mapper.Iterator()
				for it.HasNext() == types.True {
					it.Next()
				}
				if !types.IsError(it.Next()) {
					t.Error("expected exhausted error")
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := SchemalessTypedToVal(tc.val)

			if got.Type() != tc.expectedType {
				t.Errorf("expected Type %v, got %v", tc.expectedType, got.Type())
			}
			if !reflect.DeepEqual(got.Value(), tc.val) {
				t.Errorf("value mismatch: expected %v, got %v", tc.val, got.Value())
			}

			if _, err := got.ConvertToNative(tc.validNativeType); err != nil {
				t.Errorf("unexpected error converting to valid native type %v: %v", tc.validNativeType, err)
			}
			if _, err := got.ConvertToNative(tc.invalidNativeType); err == nil {
				t.Errorf("expected error converting to invalid native type %v", tc.invalidNativeType)
			}

			if got.ConvertToType(tc.expectedType) != got {
				t.Errorf("expected ConvertToType(%v) to return self", tc.expectedType)
			}
			if got.ConvertToType(types.TypeType) != tc.expectedType {
				t.Errorf("expected ConvertToType(TypeType) to return %v, got %v", tc.expectedType, got.ConvertToType(types.TypeType))
			}
			if !types.IsError(got.ConvertToType(tc.invalidTypeToConvert)) {
				t.Errorf("expected error converting to %v", tc.invalidTypeToConvert)
			}

			if tc.runTypeSpecific != nil {
				tc.runTypeSpecific(t, got)
			}
		})
	}
}

func TestSchemalessTypedStruct_Equal(t *testing.T) {
	type SimpleStruct struct {
		Name  string `json:"name"`
		Value int    `json:"value"`
	}

	type NestedStruct struct {
		Name string `json:"name"`
	}
	type ComplexStruct struct {
		Title  string       `json:"title"`
		Detail NestedStruct `json:"detail"`
		List   []string     `json:"list"`
	}

	baseStructCEL := SchemalessTypedToVal(&SimpleStruct{Name: "foo", Value: 42})
	complexBase := SchemalessTypedToVal(&ComplexStruct{
		Title:  "my complex struct",
		Detail: NestedStruct{Name: "nested foo"},
		List:   []string{"a", "b"},
	})

	tests := []struct {
		name        string
		base        ref.Val
		other       ref.Val
		expected    ref.Val
		expectedErr string
	}{
		{
			name:     "equal - same struct values",
			base:     baseStructCEL,
			other:    SchemalessTypedToVal(&SimpleStruct{Name: "foo", Value: 42}),
			expected: types.True,
		},
		{
			name:     "not equal - different struct values (same type)",
			base:     baseStructCEL,
			other:    SchemalessTypedToVal(&SimpleStruct{Name: "bar", Value: 42}),
			expected: types.False,
		},
		{
			name:     "equal - map equivalent",
			base:     baseStructCEL,
			other:    types.DefaultTypeAdapter.NativeToValue(map[string]interface{}{"name": "foo", "value": 42}),
			expected: types.True,
		},
		{
			name:     "not equal - map with different size",
			base:     baseStructCEL,
			other:    types.DefaultTypeAdapter.NativeToValue(map[string]interface{}{"name": "foo", "value": 42, "extra": "yes"}),
			expected: types.False,
		},
		{
			name:     "not equal - map with missing key (same size)",
			base:     baseStructCEL,
			other:    types.DefaultTypeAdapter.NativeToValue(map[string]interface{}{"name": "foo", "other": 42}),
			expected: types.False,
		},
		{
			name:     "not equal - map with different value",
			base:     baseStructCEL,
			other:    types.DefaultTypeAdapter.NativeToValue(map[string]interface{}{"name": "foo", "value": 43}),
			expected: types.False,
		},
		{
			name:        "not equal - incompatible type",
			base:        baseStructCEL,
			other:       types.Int(42),
			expectedErr: "no such overload",
		},
		{
			name: "equal - nested complex struct compared to map",
			base: complexBase,
			other: types.DefaultTypeAdapter.NativeToValue(map[string]interface{}{
				"title":  "my complex struct",
				"detail": map[string]interface{}{"name": "nested foo"},
				"list":   []interface{}{"a", "b"},
			}),
			expected: types.True,
		},
		{
			name: "not equal - nested complex struct compared to map (nested value mismatch)",
			base: complexBase,
			other: types.DefaultTypeAdapter.NativeToValue(map[string]interface{}{
				"title":  "my complex struct",
				"detail": map[string]interface{}{"name": "nested bar"},
				"list":   []interface{}{"a", "b"},
			}),
			expected: types.False,
		},
		{
			name: "not equal - nested complex struct compared to map (list value mismatch)",
			base: complexBase,
			other: types.DefaultTypeAdapter.NativeToValue(map[string]interface{}{
				"title":  "my complex struct",
				"detail": map[string]interface{}{"name": "nested foo"},
				"list":   []interface{}{"a", "c"},
			}),
			expected: types.False,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.base.Equal(tc.other)
			if types.IsError(got) {
				if tc.expectedErr == "" {
					t.Fatalf("unexpected error: %v", got.Value())
				}
				if got.Value().(error).Error() != tc.expectedErr {
					t.Errorf("expected error %q, got %q", tc.expectedErr, got.Value().(error).Error())
				}
				return
			}
			if tc.expectedErr != "" {
				t.Fatalf("expected error %q, got type: %v, value: %v", tc.expectedErr, got.Type(), got.Value())
			}
			if got != tc.expected {
				t.Errorf("expected equal result %v, got %v", tc.expected, got)
			}
		})
	}
}

func TestSchemalessTypedMap_Equal(t *testing.T) {
	baseMapCEL := SchemalessTypedToVal(map[string]interface{}{"a": 1, "b": "hello"})
	complexBase := SchemalessTypedToVal(map[string]interface{}{
		"title":  "nested",
		"detail": map[string]interface{}{"name": "foo"},
		"list":   []interface{}{"a", "b"},
	})

	tests := []struct {
		name        string
		base        ref.Val
		other       ref.Val
		expected    ref.Val
		expectedErr string
	}{
		{
			name:     "equal - same map values",
			base:     baseMapCEL,
			other:    SchemalessTypedToVal(map[string]interface{}{"a": 1, "b": "hello"}),
			expected: types.True,
		},
		{
			name:     "not equal - different size",
			base:     baseMapCEL,
			other:    SchemalessTypedToVal(map[string]interface{}{"a": 1, "b": "hello", "c": true}),
			expected: types.False,
		},
		{
			name:     "not equal - missing key (same size)",
			base:     baseMapCEL,
			other:    SchemalessTypedToVal(map[string]interface{}{"a": 1, "c": "hello"}),
			expected: types.False,
		},
		{
			name:     "not equal - different value (same size)",
			base:     baseMapCEL,
			other:    SchemalessTypedToVal(map[string]interface{}{"a": 1, "b": "world"}),
			expected: types.False,
		},
		{
			name:        "not equal - incompatible type",
			base:        baseMapCEL,
			other:       types.Int(42),
			expectedErr: "no such overload",
		},
		{
			name: "equal - nested complex map compared to map",
			base: complexBase,
			other: types.DefaultTypeAdapter.NativeToValue(map[string]interface{}{
				"title":  "nested",
				"detail": map[string]interface{}{"name": "foo"},
				"list":   []interface{}{"a", "b"},
			}),
			expected: types.True,
		},
		{
			name: "not equal - nested complex map compared to map (nested value mismatch)",
			base: complexBase,
			other: types.DefaultTypeAdapter.NativeToValue(map[string]interface{}{
				"title":  "nested",
				"detail": map[string]interface{}{"name": "bar"},
				"list":   []interface{}{"a", "b"},
			}),
			expected: types.False,
		},
		{
			name: "not equal - nested complex map compared to map (list value mismatch)",
			base: complexBase,
			other: types.DefaultTypeAdapter.NativeToValue(map[string]interface{}{
				"title":  "nested",
				"detail": map[string]interface{}{"name": "foo"},
				"list":   []interface{}{"a", "c"},
			}),
			expected: types.False,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.base.Equal(tc.other)
			if types.IsError(got) {
				if tc.expectedErr == "" {
					t.Fatalf("unexpected error: %v", got.Value())
				}
				if got.Value().(error).Error() != tc.expectedErr {
					t.Errorf("expected error %q, got %q", tc.expectedErr, got.Value().(error).Error())
				}
				return
			}
			if tc.expectedErr != "" {
				t.Fatalf("expected error %q, got type: %v, value: %v", tc.expectedErr, got.Type(), got.Value())
			}
			if got != tc.expected {
				t.Errorf("expected equal result %v, got %v", tc.expected, got)
			}
		})
	}
}
