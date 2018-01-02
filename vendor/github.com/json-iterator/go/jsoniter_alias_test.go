package jsoniter

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_alias(t *testing.T) {
	should := require.New(t)
	type myint int
	type myint8 int8
	type myint16 int16
	type myint32 int32
	type myint64 int64
	type myuint uint
	type myuint8 uint8
	type myuint16 uint16
	type myuint32 uint32
	type myuint64 uint64
	type myfloat32 float32
	type myfloat64 float64
	type mystring string
	type mybool bool
	type myuintptr uintptr
	var a struct {
		A myint8    `json:"a"`
		B myint16   `json:"stream"`
		C myint32   `json:"c"`
		D myint64   `json:"d"`
		E myuint8   `json:"e"`
		F myuint16  `json:"f"`
		G myuint32  `json:"g"`
		H myuint64  `json:"h"`
		I myfloat32 `json:"i"`
		J myfloat64 `json:"j"`
		K mystring  `json:"k"`
		L myint     `json:"l"`
		M myuint    `json:"m"`
		N mybool    `json:"n"`
		O myuintptr `json:"o"`
	}

	should.Nil(UnmarshalFromString(`{"a" : 1, "stream" : 1, "c": 1, "d" : 1, "e" : 1, "f" : 1, "g" : 1, "h": 1, "i" : 1, "j" : 1, "k" :"xxxx", "l" : 1, "m":1, "n": true, "o" : 1}`, &a))
	should.Equal(myfloat32(1), a.I)
	should.Equal(myfloat64(1), a.J)
	should.Equal(myint8(1), a.A)
	should.Equal(myint16(1), a.B)
	should.Equal(myint32(1), a.C)
	should.Equal(myint64(1), a.D)
	should.Equal(myuint8(1), a.E)
	should.Equal(myuint16(1), a.F)
	should.Equal(myuint32(1), a.G)
	should.Equal(myuint64(1), a.H)
	should.Equal(mystring("xxxx"), a.K)
	should.Equal(mybool(true), a.N)
	should.Equal(myuintptr(1), a.O)
	b, err := Marshal(a)
	should.Nil(err)
	should.Equal(`{"a":1,"stream":1,"c":1,"d":1,"e":1,"f":1,"g":1,"h":1,"i":1,"j":1,"k":"xxxx","l":1,"m":1,"n":true,"o":1}`, string(b))

}
