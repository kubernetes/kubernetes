// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2013, The GoGo Authors. All rights reserved.
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

package test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/gogo/protobuf/proto"
)

//http://code.google.com/p/goprotobuf/issues/detail?id=39
func TestBugUint32VarintSize(t *testing.T) {
	temp := uint32(math.MaxUint32)
	n := &NinOptNative{}
	n.Field5 = &temp
	data, err := proto.Marshal(n)
	if err != nil {
		panic(err)
	}
	if len(data) != 6 {
		t.Fatalf("data should be length 6, but its %#v", data)
	}
}

func TestBugZeroLengthSliceSize(t *testing.T) {
	n := &NinRepPackedNative{
		Field8: []int64{},
	}
	size := n.Size()
	data, err := proto.Marshal(n)
	if err != nil {
		panic(err)
	}
	if len(data) != size {
		t.Fatalf("expected %v, but got %v", len(data), size)
	}
}

//http://code.google.com/p/goprotobuf/issues/detail?id=40
func TestBugPackedProtoSize(t *testing.T) {
	n := &NinRepPackedNative{
		Field4:  []int64{172960727389894724, 2360337516664475010, 860833876131988189, 9068073014890763245, 7794843386260381831, 4023536436053141786, 8992311247496919020, 4330096163611305776, 4490411416244976467, 7873947349172707443, 2754969595834279669, 1360667855926938684, 4771480785172657389, 4875578924966668055, 8070579869808877481, 9128179594766551001, 4630419407064527516, 863844540220372892, 8208727650143073487, 7086117356301045838, 7779695211931506151, 5493835345187563535, 9119767633370806007, 9054342025895349248, 1887303228838508438, 7624573031734528281, 1874668389749611225, 3517684643468970593, 6677697606628877758, 7293473953189936168, 444475066704085538, 8594971141363049302, 1146643249094989673, 733393306232853371, 7721178528893916886, 7784452000911004429, 6436373110242711440, 6897422461738321237, 8772249155667732778, 6211871464311393541, 3061903718310406883, 7845488913176136641, 8342255034663902574, 3443058984649725748, 8410801047334832902, 7496541071517841153, 4305416923521577765, 7814967600020476457, 8671843803465481186, 3490266370361096855, 1447425664719091336, 653218597262334239, 8306243902880091940, 7851896059762409081, 5936760560798954978, 5755724498441478025, 7022701569985035966, 3707709584811468220, 529069456924666920, 7986469043681522462, 3092513330689518836, 5103541550470476202, 3577384161242626406, 3733428084624703294, 8388690542440473117, 3262468785346149388, 8788358556558007570, 5476276940198542020, 7277903243119461239, 5065861426928605020, 7533460976202697734, 1749213838654236956, 557497603941617931, 5496307611456481108, 6444547750062831720, 6992758776744205596, 7356719693428537399, 2896328872476734507, 381447079530132038, 598300737753233118, 3687980626612697715, 7240924191084283349, 8172414415307971170, 4847024388701257185, 2081764168600256551, 3394217778539123488, 6244660626429310923, 8301712215675381614, 5360615125359461174, 8410140945829785773, 3152963269026381373, 6197275282781459633, 4419829061407546410, 6262035523070047537, 2837207483933463885, 2158105736666826128, 8150764172235490711},
		Field7:  []int32{249451845, 1409974015, 393609128, 435232428, 1817529040, 91769006, 861170933, 1556185603, 1568580279, 1236375273, 512276621, 693633711, 967580535, 1950715977, 853431462, 1362390253, 159591204, 111900629, 322985263, 279671129, 1592548430, 465651370, 733849989, 1172059400, 1574824441, 263541092, 1271612397, 1520584358, 467078791, 117698716, 1098255064, 2054264846, 1766452305, 1267576395, 1557505617, 1187833560, 956187431, 1970977586, 1160235159, 1610259028, 489585797, 459139078, 566263183, 954319278, 1545018565, 1753946743, 948214318, 422878159, 883926576, 1424009347, 824732372, 1290433180, 80297942, 417294230, 1402647904, 2078392782, 220505045, 787368129, 463781454, 293083578, 808156928, 293976361},
		Field9:  []uint32{0xaa4976e8, 0x3da8cc4c, 0x8c470d83, 0x344d964e, 0x5b90925, 0xa4c4d34e, 0x666eff19, 0xc238e552, 0x9be53bb6, 0x56364245, 0x33ee079d, 0x96bf0ede, 0x7941b74f, 0xdb07cb47, 0x6d76d827, 0x9b211d5d, 0x2798adb6, 0xe48b0c3b, 0x87061b21, 0x48f4e4d2, 0x3e5d5c12, 0x5ee91288, 0x336d4f35, 0xe1d44941, 0xc065548d, 0x2953d73f, 0x873af451, 0xfc769db, 0x9f1bf8da, 0x9baafdfc, 0xf1d3d770, 0x5bb5d2b4, 0xc2c67c48, 0x6845c4c1, 0xa48f32b0, 0xbb04bb70, 0xa5b1ca36, 0x8d98356a, 0x2171f654, 0x5ae279b0, 0x6c4a3d6b, 0x4fff5468, 0xcf9bf851, 0x68513614, 0xdbecd9b0, 0x9553ed3c, 0xa494a736, 0x42205438, 0xbf8e5caa, 0xd3283c6, 0x76d20788, 0x9179826f, 0x96b24f85, 0xbc2eacf4, 0xe4afae0b, 0x4bca85cb, 0x35e63b5b, 0xd7ccee0c, 0x2b506bb9, 0xe78e9f44, 0x9ad232f1, 0x99a37335, 0xa5d6ffc8},
		Field11: []uint64{0x53c01ebc, 0x4fb85ba6, 0x8805eea1, 0xb20ec896, 0x93b63410, 0xec7c9492, 0x50765a28, 0x19592106, 0x2ecc59b3, 0x39cd474f, 0xe4c9e47, 0x444f48c5, 0xe7731d32, 0xf3f43975, 0x603caedd, 0xbb05a1af, 0xa808e34e, 0x88580b07, 0x4c96bbd1, 0x730b4ab9, 0xed126e2b, 0x6db48205, 0x154ba1b9, 0xc26bfb6a, 0x389aa052, 0x869d966c, 0x7c86b366, 0xcc8edbcd, 0xfa8d6dad, 0xcf5857d9, 0x2d9cda0f, 0x1218a0b8, 0x41bf997, 0xf0ca65ac, 0xa610d4b9, 0x8d362e28, 0xb7212d87, 0x8e0fe109, 0xbee041d9, 0x759be2f6, 0x35fef4f3, 0xaeacdb71, 0x10888852, 0xf4e28117, 0xe2a14812, 0x73b748dc, 0xd1c3c6b2, 0xfef41bf0, 0xc9b43b62, 0x810e4faa, 0xcaa41c06, 0x1893fe0d, 0xedc7c850, 0xd12b9eaa, 0x467ee1a9, 0xbe84756b, 0xda7b1680, 0xdc069ffe, 0xf1e7e9f9, 0xb3d95370, 0xa92b77df, 0x5693ac41, 0xd04b7287, 0x27aebf15, 0x837b316e, 0x4dbe2263, 0xbab70c67, 0x547dab21, 0x3c346c1f, 0xb8ef0e4e, 0xfe2d03ce, 0xe1d75955, 0xfec1306, 0xba35c23e, 0xb784ed04, 0x2a4e33aa, 0x7e19d09a, 0x3827c1fe, 0xf3a51561, 0xef765e2b, 0xb044256c, 0x62b322be, 0xf34d56be, 0xeb71b369, 0xffe1294f, 0x237fe8d0, 0x77a1473b, 0x239e1196, 0xdd19bf3d, 0x82c91fe1, 0x95361c57, 0xffea3f1b, 0x1a094c84},
		Field12: []int64{8308420747267165049, 3664160795077875961, 7868970059161834817, 7237335984251173739, 5254748003907196506, 3362259627111837480, 430460752854552122, 5119635556501066533, 1277716037866233522, 9185775384759813768, 833932430882717888, 7986528304451297640, 6792233378368656337, 2074207091120609721, 1788723326198279432, 7756514594746453657, 2283775964901597324, 3061497730110517191, 7733947890656120277, 626967303632386244, 7822928600388582821, 3489658753000061230, 168869995163005961, 248814782163480763, 477885608911386247, 4198422415674133867, 3379354662797976109, 9925112544736939, 1486335136459138480, 4561560414032850671, 1010864164014091267, 186722821683803084, 5106357936724819318, 1298160820191228988, 4675403242419953145, 7130634540106489752, 7101280006672440929, 7176058292431955718, 9109875054097770321, 6810974877085322872, 4736707874303993641, 8993135362721382187, 6857881554990254283, 3704748883307461680, 1099360832887634994, 5207691918707192633, 5984721695043995243},
	}
	size := proto.Size(n)
	data, err := proto.Marshal(n)
	if err != nil {
		panic(err)
	}
	if len(data) != size {
		t.Fatalf("expected %v, but got %v diff is %v", len(data), size, len(data)-size)
	}
}

func testSize(m interface {
	proto.Message
	Size() int
}, desc string, expected int) ([]byte, error) {
	data, err := proto.Marshal(m)
	if err != nil {
		return nil, err
	}
	protoSize := proto.Size(m)
	mSize := m.Size()
	lenData := len(data)
	if protoSize != mSize || protoSize != lenData || mSize != lenData {
		return nil, fmt.Errorf("%s proto.Size(m){%d} != m.Size(){%d} != len(data){%d}", desc, protoSize, mSize, lenData)
	}
	if got := protoSize; got != expected {
		return nil, fmt.Errorf("%s proto.Size(m) got %d expected %d", desc, got, expected)
	}
	if got := mSize; got != expected {
		return nil, fmt.Errorf("%s m.Size() got %d expected %d", desc, got, expected)
	}
	if got := lenData; got != expected {
		return nil, fmt.Errorf("%s len(data) got %d expected %d", desc, got, expected)
	}
	return data, nil
}

func TestInt32Int64Compatibility(t *testing.T) {

	//test nullable int32 and int64

	data1, err := testSize(&NinOptNative{
		Field3: proto.Int32(-1),
	}, "nullable", 11)
	if err != nil {
		t.Error(err)
	}
	//change marshaled data1 to unmarshal into 4th field which is an int64
	data1[0] = uint8(uint32(4 /*fieldNumber*/)<<3 | uint32(0 /*wireType*/))
	u1 := &NinOptNative{}
	if err = proto.Unmarshal(data1, u1); err != nil {
		t.Error(err)
	}
	if !u1.Equal(&NinOptNative{
		Field4: proto.Int64(-1),
	}) {
		t.Error("nullable unmarshaled int32 is not the same int64")
	}

	//test non-nullable int32 and int64

	data2, err := testSize(&NidOptNative{
		Field3: -1,
	}, "non nullable", 67)
	if err != nil {
		t.Error(err)
	}
	//change marshaled data2 to unmarshal into 4th field which is an int64
	field3 := uint8(uint32(3 /*fieldNumber*/)<<3 | uint32(0 /*wireType*/))
	field4 := uint8(uint32(4 /*fieldNumber*/)<<3 | uint32(0 /*wireType*/))
	for i, c := range data2 {
		if c == field4 {
			data2[i] = field3
		} else if c == field3 {
			data2[i] = field4
		}
	}
	u2 := &NidOptNative{}
	if err = proto.Unmarshal(data2, u2); err != nil {
		t.Error(err)
	}
	if !u2.Equal(&NidOptNative{
		Field4: -1,
	}) {
		t.Error("non nullable unmarshaled int32 is not the same int64")
	}

	//test packed repeated int32 and int64

	m4 := &NinRepPackedNative{
		Field3: []int32{-1},
	}
	data4, err := testSize(m4, "packed", 12)
	if err != nil {
		t.Error(err)
	}
	u4 := &NinRepPackedNative{}
	if err := proto.Unmarshal(data4, u4); err != nil {
		t.Error(err)
	}
	if err := u4.VerboseEqual(m4); err != nil {
		t.Fatalf("%#v", u4)
	}

	//test repeated int32 and int64

	if _, err := testSize(&NinRepNative{
		Field3: []int32{-1},
	}, "repeated", 11); err != nil {
		t.Error(err)
	}

	t.Logf("tested all")
}

func TestRepeatedExtensionsMsgsIssue161(t *testing.T) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	rep := 10
	nins := make([]*NinOptNative, rep)
	for i := range nins {
		nins[i] = NewPopulatedNinOptNative(r, true)
	}
	input := &MyExtendable{}
	if err := proto.SetExtension(input, E_FieldE, nins); err != nil {
		t.Fatal(err)
	}
	data, err := proto.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	output := &MyExtendable{}
	if err := proto.Unmarshal(data, output); err != nil {
		t.Fatal(err)
	}
	if !input.Equal(output) {
		t.Fatalf("want %#v but got %#v", input, output)
	}
	data2, err2 := proto.Marshal(output)
	if err2 != nil {
		t.Fatal(err2)
	}
	if len(data) != len(data2) {
		t.Fatal("expected equal length buffers")
	}
}

func TestRepeatedExtensionsFieldsIssue161(t *testing.T) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	rep := 10
	ints := make([]int64, rep)
	for i := range ints {
		ints[i] = r.Int63()
	}
	input := &MyExtendable{}
	if err := proto.SetExtension(input, E_FieldD, ints); err != nil {
		t.Fatal(err)
	}
	data, err := proto.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	output := &MyExtendable{}
	if err := proto.Unmarshal(data, output); err != nil {
		t.Fatal(err)
	}
	if !input.Equal(output) {
		t.Fatalf("want %#v but got %#v", input, output)
	}
	data2, err2 := proto.Marshal(output)
	if err2 != nil {
		t.Fatal(err2)
	}
	if len(data) != len(data2) {
		t.Fatal("expected equal length buffers")
	}
}
