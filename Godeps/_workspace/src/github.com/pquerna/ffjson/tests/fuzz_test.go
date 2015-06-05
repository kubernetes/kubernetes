/**
 *  Copyright 2014 Paul Querna, Klaus Post
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package tff

import (
	"encoding/json"
	fuzz "github.com/google/gofuzz"
	"github.com/stretchr/testify/require"
	"math/rand"
	"runtime"
	"strings"
	"testing"
	"time"
)

type Fuzz struct {
	A uint8
	B uint16
	C uint32
	D uint64

	E int8
	F int16
	G int32
	H int64

	I float32
	J float64

	M byte
	N rune

	O int
	P uint
	Q string
	R bool
	S time.Time

	Ap *uint8
	Bp *uint16
	Cp *uint32
	Dp *uint64

	Ep *int8
	Fp *int16
	Gp *int32
	Hp *int64

	Ip *float32
	Jp *float64

	Mp *byte
	Np *rune

	Op *int
	Pp *uint
	Qp *string
	Rp *bool
	Sp *time.Time

	Aa []uint8
	Ba []uint16
	Ca []uint32
	Da []uint64

	Ea []int8
	Fa []int16
	Ga []int32
	Ha []int64

	Ia []float32
	Ja []float64

	Ma []byte
	Na []rune

	Oa []int
	Pa []uint
	Qa []string
	Ra []bool

	Aap []*uint8
	Bap []*uint16
	Cap []*uint32
	Dap []*uint64

	Eap []*int8
	Fap []*int16
	Gap []*int32
	Hap []*int64

	Iap []*float32
	Jap []*float64

	Map []*byte
	Nap []*rune

	Oap []*int
	Pap []*uint
	Qap []*string
	Rap []*bool
}

// Return a time no later than 5000 years from unix datum.
// JSON cannot handle dates after year 9999.
func fuzzTime(t *time.Time, c fuzz.Continue) {
	sec := c.Rand.Int63()
	nsec := c.Rand.Int63()
	// No more than 5000 years in the future
	sec %= 5000 * 365 * 24 * 60 * 60
	*t = time.Unix(sec, nsec)
}

func fuzzTimeArray(t *[]time.Time, c fuzz.Continue) {
	var i uint64
	rv := make([]time.Time, 0)
	count := c.RandUint64() % 50
	for i = 0; i < count; i++ {
		var tmp time.Time
		fuzzTime(&tmp, c)
		rv = append(rv, tmp)
	}
	*t = rv
}

// Test 1000 iterations
func TestFuzzCycle(t *testing.T) {
	f := fuzz.New()
	f.NumElements(0, 50)
	f.NilChance(0.1)
	f.Funcs(fuzzTime)

	rFF := FfFuzz{}
	r := Fuzz{}
	for i := 0; i < 1000; i++ {
		if true || i > 0 {
			// TODO: Re-enable after fixing:
			// https://github.com/pquerna/ffjson/issues/82
			f.RandSource(rand.New(rand.NewSource(int64(i * 324221))))
			f.Fuzz(&r)
		}
		rFF.A = r.A
		rFF.B = r.B
		rFF.C = r.C
		rFF.D = r.D
		rFF.E = r.E
		rFF.F = r.F
		rFF.G = r.G
		rFF.H = r.H
		rFF.I = r.I
		rFF.J = r.J
		rFF.M = r.M
		rFF.N = r.N
		rFF.O = r.O
		rFF.P = r.P
		rFF.Q = r.Q
		rFF.R = r.R
		rFF.S = r.S

		rFF.Ap = r.Ap
		rFF.Bp = r.Bp
		rFF.Cp = r.Cp
		rFF.Dp = r.Dp
		rFF.Ep = r.Ep
		rFF.Fp = r.Fp
		rFF.Gp = r.Gp
		rFF.Hp = r.Hp
		rFF.Ip = r.Ip
		rFF.Jp = r.Jp
		rFF.Mp = r.Mp
		rFF.Np = r.Np
		rFF.Op = r.Op
		rFF.Pp = r.Pp
		rFF.Qp = r.Qp
		rFF.Rp = r.Rp
		rFF.Sp = r.Sp

		rFF.Aa = r.Aa
		rFF.Ba = r.Ba
		rFF.Ca = r.Ca
		rFF.Da = r.Da
		rFF.Ea = r.Ea
		rFF.Fa = r.Fa
		rFF.Ga = r.Ga
		rFF.Ha = r.Ha
		rFF.Ia = r.Ia
		rFF.Ja = r.Ja
		rFF.Ma = r.Ma
		rFF.Na = r.Na
		rFF.Oa = r.Oa
		rFF.Pa = r.Pa
		rFF.Qa = r.Qa
		rFF.Ra = r.Ra

		rFF.Aap = r.Aap
		rFF.Bap = r.Bap
		rFF.Cap = r.Cap
		rFF.Dap = r.Dap
		rFF.Eap = r.Eap
		rFF.Fap = r.Fap
		rFF.Gap = r.Gap
		rFF.Hap = r.Hap
		rFF.Iap = r.Iap
		rFF.Jap = r.Jap
		rFF.Map = r.Map
		rFF.Nap = r.Nap
		rFF.Oap = r.Oap
		rFF.Pap = r.Pap
		rFF.Qap = r.Qap
		rFF.Rap = r.Rap
		testSameMarshal(t, &r, &rFF)
		testCycle(t, &r, &rFF)
	}
}

// Test 1000 iterations
func TestFuzzOmitCycle(t *testing.T) {
	f := fuzz.New()
	f.NumElements(0, 10)
	f.NilChance(0.5)
	f.Funcs(fuzzTime)

	rFF := FfFuzzOmitEmpty{}
	r := FuzzOmitEmpty{}
	for i := 0; i <= 1000; i++ {
		if i > 0 {
			f.RandSource(rand.New(rand.NewSource(int64(i * 324221))))
			f.Fuzz(&r)
		}
		rFF.A = r.A
		rFF.B = r.B
		rFF.C = r.C
		rFF.D = r.D
		rFF.E = r.E
		rFF.F = r.F
		rFF.G = r.G
		rFF.H = r.H
		rFF.I = r.I
		rFF.J = r.J
		rFF.M = r.M
		rFF.N = r.N
		rFF.O = r.O
		rFF.P = r.P
		rFF.Q = r.Q
		rFF.R = r.R
		rFF.S = r.S

		rFF.Ap = r.Ap
		rFF.Bp = r.Bp
		rFF.Cp = r.Cp
		rFF.Dp = r.Dp
		rFF.Ep = r.Ep
		rFF.Fp = r.Fp
		rFF.Gp = r.Gp
		rFF.Hp = r.Hp
		rFF.Ip = r.Ip
		rFF.Jp = r.Jp
		rFF.Mp = r.Mp
		rFF.Np = r.Np
		rFF.Op = r.Op
		rFF.Pp = r.Pp
		rFF.Qp = r.Qp
		rFF.Rp = r.Rp
		rFF.Sp = r.Sp

		rFF.Aa = r.Aa
		rFF.Ba = r.Ba
		rFF.Ca = r.Ca
		rFF.Da = r.Da
		rFF.Ea = r.Ea
		rFF.Fa = r.Fa
		rFF.Ga = r.Ga
		rFF.Ha = r.Ha
		rFF.Ia = r.Ia
		rFF.Ja = r.Ja
		rFF.Ma = r.Ma
		rFF.Na = r.Na
		rFF.Oa = r.Oa
		rFF.Pa = r.Pa
		rFF.Qa = r.Qa
		rFF.Ra = r.Ra

		rFF.Aap = r.Aap
		rFF.Bap = r.Bap
		rFF.Cap = r.Cap
		rFF.Dap = r.Dap
		rFF.Eap = r.Eap
		rFF.Fap = r.Fap
		rFF.Gap = r.Gap
		rFF.Hap = r.Hap
		rFF.Iap = r.Iap
		rFF.Jap = r.Jap
		rFF.Map = r.Map
		rFF.Nap = r.Nap
		rFF.Oap = r.Oap
		rFF.Pap = r.Pap
		rFF.Qap = r.Qap
		rFF.Rap = r.Rap
		testSameMarshal(t, &r, &rFF)
		testCycle(t, &r, &rFF)
	}
}

// Test 1000 iterations
func TestFuzzStringCycle(t *testing.T) {
	ver := runtime.Version()
	if strings.Contains(ver, "go1.3") || strings.Contains(ver, "go1.2") {
		t.Skipf("Test requires go v1.4 or later, this is %s", ver)
	}
	f := fuzz.New()
	f.NumElements(0, 50)
	f.NilChance(0.1)
	f.Funcs(fuzzTime)

	rFF := FfFuzzString{}
	r := FuzzString{}
	for i := 0; i < 1000; i++ {
		if i > 0 {
			f.RandSource(rand.New(rand.NewSource(int64(i * 324221))))
			f.Fuzz(&r)
		}
		rFF.A = r.A
		rFF.B = r.B
		rFF.C = r.C
		rFF.D = r.D
		rFF.E = r.E
		rFF.F = r.F
		rFF.G = r.G
		rFF.H = r.H
		rFF.I = r.I
		rFF.J = r.J
		rFF.M = r.M
		rFF.N = r.N
		rFF.O = r.O
		rFF.P = r.P
		rFF.Q = r.Q
		rFF.R = r.R

		// https://github.com/golang/go/issues/9812
		// rFF.S = r.S

		rFF.Ap = r.Ap
		rFF.Bp = r.Bp
		rFF.Cp = r.Cp
		rFF.Dp = r.Dp
		rFF.Ep = r.Ep
		rFF.Fp = r.Fp
		rFF.Gp = r.Gp
		rFF.Hp = r.Hp
		rFF.Ip = r.Ip
		rFF.Jp = r.Jp
		rFF.Mp = r.Mp
		rFF.Np = r.Np
		rFF.Op = r.Op
		rFF.Pp = r.Pp
		rFF.Qp = r.Qp
		rFF.Rp = r.Rp
		// https://github.com/golang/go/issues/9812
		// rFF.Sp = r.Sp

		// The "string" option signals that a field is stored as JSON inside a JSON-encoded string. It applies only to fields of string, floating point, or integer types. This extra level of encoding is sometimes used when communicating with JavaScript programs.
		// Therefore tests on byte arrays are removed, since the golang decoder chokes on them.
		testSameMarshal(t, &r, &rFF)

		// Disabled: https://github.com/pquerna/ffjson/issues/80
		// testCycle(t, &r, &rFF)
	}
}

// Fuzz test for 1000 iterations
func testTypeFuzz(t *testing.T, base interface{}, ff interface{}) {
	testTypeFuzzN(t, base, ff, 1000)
}

// Fuzz test for N iterations
func testTypeFuzzN(t *testing.T, base interface{}, ff interface{}, n int) {
	require.Implements(t, (*json.Marshaler)(nil), ff)
	require.Implements(t, (*json.Unmarshaler)(nil), ff)
	require.Implements(t, (*marshalerFaster)(nil), ff)
	require.Implements(t, (*unmarshalFaster)(nil), ff)

	if _, ok := base.(unmarshalFaster); ok {
		require.FailNow(t, "base should not have a UnmarshalJSONFFLexer")
	}

	if _, ok := base.(marshalerFaster); ok {
		require.FailNow(t, "base should not have a MarshalJSONBuf")
	}

	f := fuzz.New()
	f.NumElements(0, 1+n/40)
	f.NilChance(0.2)
	f.Funcs(fuzzTime, fuzzTimeArray)
	for i := 0; i < n; i++ {
		f.RandSource(rand.New(rand.NewSource(int64(i * 5275))))
		f.Fuzz(base)
		f.RandSource(rand.New(rand.NewSource(int64(i * 5275))))
		f.Fuzz(ff)

		testSameMarshal(t, base, ff)
		testCycle(t, base, ff)
	}
}

func TestFuzzArray(t *testing.T) {
	testTypeFuzz(t, &Tarray{X: []int{}}, &Xarray{X: []int{}})
}

func TestFuzzArrayPtr(t *testing.T) {
	testTypeFuzz(t, &TarrayPtr{X: []*int{}}, &XarrayPtr{X: []*int{}})
}

func TestFuzzTimeDuration(t *testing.T) {
	testTypeFuzz(t, &Tduration{}, &Xduration{})
}

func TestFuzzBool(t *testing.T) {
	testTypeFuzz(t, &Tbool{}, &Xbool{})
}

func TestFuzzInt(t *testing.T) {
	testTypeFuzz(t, &Tint{}, &Xint{})
}

func TestFuzzByte(t *testing.T) {
	testTypeFuzz(t, &Tbyte{}, &Xbyte{})
}

func TestFuzzInt8(t *testing.T) {
	testTypeFuzz(t, &Tint8{}, &Xint8{})
}

func TestFuzzInt16(t *testing.T) {
	testTypeFuzz(t, &Tint16{}, &Xint16{})
}

func TestFuzzInt32(t *testing.T) {
	testTypeFuzz(t, &Tint32{}, &Xint32{})
}

func TestFuzzInt64(t *testing.T) {
	testTypeFuzz(t, &Tint64{}, &Xint64{})
}

func TestFuzzUint(t *testing.T) {
	testTypeFuzz(t, &Tuint{}, &Xuint{})
}

func TestFuzzUint8(t *testing.T) {
	testTypeFuzz(t, &Tuint8{}, &Xuint8{})
}

func TestFuzzUint16(t *testing.T) {
	testTypeFuzz(t, &Tuint16{}, &Xuint16{})
}

func TestFuzzUint32(t *testing.T) {
	testTypeFuzz(t, &Tuint32{}, &Xuint32{})
}

func TestFuzzUint64(t *testing.T) {
	testTypeFuzz(t, &Tuint64{}, &Xuint64{})
}

func TestFuzzUintptr(t *testing.T) {
	testTypeFuzz(t, &Tuintptr{}, &Xuintptr{})
}

func TestFuzzFloat32(t *testing.T) {
	testTypeFuzz(t, &Tfloat32{}, &Xfloat32{})
}

func TestFuzzFloat64(t *testing.T) {
	testTypeFuzz(t, &Tfloat64{}, &Xfloat64{})
}

func TestFuzzString(t *testing.T) {
	testTypeFuzz(t, &Tstring{}, &Xstring{})
	testTypeFuzz(t, &Tmystring{}, &Xmystring{})
	testTypeFuzz(t, &TmystringPtr{}, &XmystringPtr{})
}

func TestFuzzArrayTimeDuration(t *testing.T) {
	testTypeFuzz(t, &ATduration{}, &AXduration{})
}

func TestFuzzArrayBool(t *testing.T) {
	testTypeFuzz(t, &ATbool{}, &AXbool{})
}

func TestFuzzArrayInt(t *testing.T) {
	testTypeFuzz(t, &ATint{}, &AXint{})
}

func TestFuzzArrayByte(t *testing.T) {
	testTypeFuzz(t, &ATbyte{}, &AXbyte{})
}

func TestFuzzArrayInt8(t *testing.T) {
	testTypeFuzz(t, &ATint8{}, &AXint8{})
}

func TestFuzzArrayInt16(t *testing.T) {
	testTypeFuzz(t, &ATint16{}, &AXint16{})
}

func TestFuzzArrayInt32(t *testing.T) {
	testTypeFuzz(t, &ATint32{}, &AXint32{})
}

func TestFuzzArrayInt64(t *testing.T) {
	testTypeFuzz(t, &ATint64{}, &AXint64{})
}

func TestFuzzArrayUint(t *testing.T) {
	testTypeFuzz(t, &ATuint{}, &AXuint{})
}

func TestFuzzArrayUint8(t *testing.T) {
	testTypeFuzz(t, &ATuint8{}, &AXuint8{})
}

func TestFuzzArrayUint16(t *testing.T) {
	testTypeFuzz(t, &ATuint16{}, &AXuint16{})
}

func TestFuzzArrayUint32(t *testing.T) {
	testTypeFuzz(t, &ATuint32{}, &AXuint32{})
}

func TestFuzzArrayUint64(t *testing.T) {
	testTypeFuzz(t, &ATuint64{}, &AXuint64{})
}

func TestFuzzArrayUintptr(t *testing.T) {
	testTypeFuzz(t, &ATuintptr{}, &AXuintptr{})
}

func TestFuzzArrayFloat32(t *testing.T) {
	testTypeFuzz(t, &ATfloat32{}, &AXfloat32{})
}

func TestFuzzArrayFloat64(t *testing.T) {
	testTypeFuzz(t, &ATfloat64{}, &AXfloat64{})
}

func TestFuzzArrayTime(t *testing.T) {
	testTypeFuzz(t, &ATtime{}, &AXtime{})
}

func TestFuzzI18nName(t *testing.T) {
	testTypeFuzz(t, &TI18nName{}, &XI18nName{})
}

func TestFuzzInlineStructs(t *testing.T) {
	testTypeFuzzN(t, &TInlineStructs{}, &XInlineStructs{}, 100)
}

// This contains maps.
// Since map order is random, we can expect the encoding order to be random
// Therefore we cannot use binary compare.
func TestFuzzMapToType(t *testing.T) {
	base := &TTestMaps{}
	ff := &XTestMaps{}
	f := fuzz.New()
	f.NumElements(0, 50)
	f.NilChance(0.1)
	f.Funcs(fuzzTime)
	for i := 0; i < 100; i++ {
		f.RandSource(rand.New(rand.NewSource(int64(i * 5275))))
		f.Fuzz(base)
		ff = &XTestMaps{*base}

		bufbase, err := json.Marshal(base)
		require.NoError(t, err, "base[%T] failed to Marshal", base)

		bufff, err := json.Marshal(ff)
		require.NoError(t, err, "ff[%T] failed to Marshal", ff)

		var baseD map[string]interface{}
		var ffD map[string]interface{}

		err = json.Unmarshal(bufbase, &baseD)
		require.NoError(t, err, "ff[%T] failed to Unmarshal", base)

		err = json.Unmarshal(bufff, &ffD)
		require.NoError(t, err, "ff[%T] failed to Unmarshal", ff)

		require.Equal(t, baseD, ffD, "Inspected struct difference of base[%T] != ff[%T]", base, ff)
	}
}

func TestFuzzReType(t *testing.T) {
	testTypeFuzzN(t, &TReTyped{}, &XReTyped{}, 100)
}
