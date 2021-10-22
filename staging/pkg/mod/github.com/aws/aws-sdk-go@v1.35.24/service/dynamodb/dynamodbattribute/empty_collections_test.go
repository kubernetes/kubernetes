package dynamodbattribute

import (
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/dynamodb"
)

type testEmptyCollectionsNumericalScalars struct {
	String string

	Uint8  uint8
	Uint16 uint16
	Uint32 uint32
	Uint64 uint64

	Int8  int8
	Int16 int16
	Int32 int32
	Int64 int64

	Float32 float32
	Float64 float64
}

type testEmptyCollectionsOmittedNumericalScalars struct {
	String string `dynamodbav:",omitempty"`

	Uint8  uint8  `dynamodbav:",omitempty"`
	Uint16 uint16 `dynamodbav:",omitempty"`
	Uint32 uint32 `dynamodbav:",omitempty"`
	Uint64 uint64 `dynamodbav:",omitempty"`

	Int8  int8  `dynamodbav:",omitempty"`
	Int16 int16 `dynamodbav:",omitempty"`
	Int32 int32 `dynamodbav:",omitempty"`
	Int64 int64 `dynamodbav:",omitempty"`

	Float32 float32 `dynamodbav:",omitempty"`
	Float64 float64 `dynamodbav:",omitempty"`
}

type testEmptyCollectionsPtrScalars struct {
	PtrString *string

	PtrUint8  *uint8
	PtrUint16 *uint16
	PtrUint32 *uint32
	PtrUint64 *uint64

	PtrInt8  *int8
	PtrInt16 *int16
	PtrInt32 *int32
	PtrInt64 *int64

	PtrFloat32 *float32
	PtrFloat64 *float64
}

type testEmptyCollectionsOmittedPtrNumericalScalars struct {
	PtrUint8  *uint8  `dynamodbav:",omitempty"`
	PtrUint16 *uint16 `dynamodbav:",omitempty"`
	PtrUint32 *uint32 `dynamodbav:",omitempty"`
	PtrUint64 *uint64 `dynamodbav:",omitempty"`

	PtrInt8  *int8  `dynamodbav:",omitempty"`
	PtrInt16 *int16 `dynamodbav:",omitempty"`
	PtrInt32 *int32 `dynamodbav:",omitempty"`
	PtrInt64 *int64 `dynamodbav:",omitempty"`

	PtrFloat32 *float32 `dynamodbav:",omitempty"`
	PtrFloat64 *float64 `dynamodbav:",omitempty"`
}

type testEmptyCollectionTypes struct {
	Map       map[string]string
	Slice     []string
	ByteSlice []byte
	ByteArray [4]byte
	ZeroArray [0]byte
	BinarySet [][]byte `dynamodbav:",binaryset"`
	NumberSet []int    `dynamodbav:",numberset"`
	StringSet []string `dynamodbav:",stringset"`
}

type testEmptyCollectionTypesOmitted struct {
	Map       map[string]string `dynamodbav:",omitempty"`
	Slice     []string          `dynamodbav:",omitempty"`
	ByteSlice []byte            `dynamodbav:",omitempty"`
	ByteArray [4]byte           `dynamodbav:",omitempty"`
	ZeroArray [0]byte           `dynamodbav:",omitempty"`
	BinarySet [][]byte          `dynamodbav:",binaryset,omitempty"`
	NumberSet []int             `dynamodbav:",numberset,omitempty"`
	StringSet []string          `dynamodbav:",stringset,omitempty"`
}

type testEmptyCollectionStruct struct {
	Int int
}

type testEmptyCollectionStructOmitted struct {
	Slice []string `dynamodbav:",omitempty"`
}

var sharedEmptyCollectionsTestCases = []struct {
	encoderOpts      func(encoder *Encoder)
	in               *dynamodb.AttributeValue
	actual, expected interface{}
	err              error
}{
	// scalars with zero value
	0: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"String":  {NULL: aws.Bool(true)},
				"Uint8":   {N: aws.String("0")},
				"Uint16":  {N: aws.String("0")},
				"Uint32":  {N: aws.String("0")},
				"Uint64":  {N: aws.String("0")},
				"Int8":    {N: aws.String("0")},
				"Int16":   {N: aws.String("0")},
				"Int32":   {N: aws.String("0")},
				"Int64":   {N: aws.String("0")},
				"Float32": {N: aws.String("0")},
				"Float64": {N: aws.String("0")},
			},
		},
		actual:   &testEmptyCollectionsNumericalScalars{},
		expected: testEmptyCollectionsNumericalScalars{},
	},
	// scalars with non-zero values
	1: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"String":  {S: aws.String("test string")},
				"Uint8":   {N: aws.String("1")},
				"Uint16":  {N: aws.String("2")},
				"Uint32":  {N: aws.String("3")},
				"Uint64":  {N: aws.String("4")},
				"Int8":    {N: aws.String("-5")},
				"Int16":   {N: aws.String("-6")},
				"Int32":   {N: aws.String("-7")},
				"Int64":   {N: aws.String("-8")},
				"Float32": {N: aws.String("9.9")},
				"Float64": {N: aws.String("10.1")},
			},
		},
		actual: &testEmptyCollectionsNumericalScalars{},
		expected: testEmptyCollectionsNumericalScalars{
			String:  "test string",
			Uint8:   1,
			Uint16:  2,
			Uint32:  3,
			Uint64:  4,
			Int8:    -5,
			Int16:   -6,
			Int32:   -7,
			Int64:   -8,
			Float32: 9.9,
			Float64: 10.1,
		},
	},
	// omittable scalars with zero value
	2: {
		in:       &dynamodb.AttributeValue{M: map[string]*dynamodb.AttributeValue{}},
		actual:   &testEmptyCollectionsOmittedNumericalScalars{},
		expected: testEmptyCollectionsOmittedNumericalScalars{},
	},
	// omittable scalars with non-zero value
	3: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"String":  {S: aws.String("test string")},
				"Uint8":   {N: aws.String("1")},
				"Uint16":  {N: aws.String("2")},
				"Uint32":  {N: aws.String("3")},
				"Uint64":  {N: aws.String("4")},
				"Int8":    {N: aws.String("-5")},
				"Int16":   {N: aws.String("-6")},
				"Int32":   {N: aws.String("-7")},
				"Int64":   {N: aws.String("-8")},
				"Float32": {N: aws.String("9.9")},
				"Float64": {N: aws.String("10.1")},
			},
		},
		actual: &testEmptyCollectionsOmittedNumericalScalars{},
		expected: testEmptyCollectionsOmittedNumericalScalars{
			String:  "test string",
			Uint8:   1,
			Uint16:  2,
			Uint32:  3,
			Uint64:  4,
			Int8:    -5,
			Int16:   -6,
			Int32:   -7,
			Int64:   -8,
			Float32: 9.9,
			Float64: 10.1,
		},
	},
	// nil pointer scalars
	4: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"PtrString":  {NULL: aws.Bool(true)},
				"PtrUint8":   {NULL: aws.Bool(true)},
				"PtrUint16":  {NULL: aws.Bool(true)},
				"PtrUint32":  {NULL: aws.Bool(true)},
				"PtrUint64":  {NULL: aws.Bool(true)},
				"PtrInt8":    {NULL: aws.Bool(true)},
				"PtrInt16":   {NULL: aws.Bool(true)},
				"PtrInt32":   {NULL: aws.Bool(true)},
				"PtrInt64":   {NULL: aws.Bool(true)},
				"PtrFloat32": {NULL: aws.Bool(true)},
				"PtrFloat64": {NULL: aws.Bool(true)},
			},
		},
		actual:   &testEmptyCollectionsPtrScalars{},
		expected: testEmptyCollectionsPtrScalars{},
	},
	// non-nil pointer to scalars with zero value
	5: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"PtrString":  {NULL: aws.Bool(true)},
				"PtrUint8":   {N: aws.String("0")},
				"PtrUint16":  {N: aws.String("0")},
				"PtrUint32":  {N: aws.String("0")},
				"PtrUint64":  {N: aws.String("0")},
				"PtrInt8":    {N: aws.String("0")},
				"PtrInt16":   {N: aws.String("0")},
				"PtrInt32":   {N: aws.String("0")},
				"PtrInt64":   {N: aws.String("0")},
				"PtrFloat32": {N: aws.String("0")},
				"PtrFloat64": {N: aws.String("0")},
			},
		},
		actual: &testEmptyCollectionsPtrScalars{},
		expected: testEmptyCollectionsPtrScalars{
			PtrUint8:   aws.Uint8(0),
			PtrUint16:  aws.Uint16(0),
			PtrUint32:  aws.Uint32(0),
			PtrUint64:  aws.Uint64(0),
			PtrInt8:    aws.Int8(0),
			PtrInt16:   aws.Int16(0),
			PtrInt32:   aws.Int32(0),
			PtrInt64:   aws.Int64(0),
			PtrFloat32: aws.Float32(0),
			PtrFloat64: aws.Float64(0),
		},
	},
	// pointer scalars non-nil non-zero
	6: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"PtrString":  {S: aws.String("test string")},
				"PtrUint8":   {N: aws.String("1")},
				"PtrUint16":  {N: aws.String("2")},
				"PtrUint32":  {N: aws.String("3")},
				"PtrUint64":  {N: aws.String("4")},
				"PtrInt8":    {N: aws.String("-5")},
				"PtrInt16":   {N: aws.String("-6")},
				"PtrInt32":   {N: aws.String("-7")},
				"PtrInt64":   {N: aws.String("-8")},
				"PtrFloat32": {N: aws.String("9.9")},
				"PtrFloat64": {N: aws.String("10.1")},
			},
		},
		actual: &testEmptyCollectionsPtrScalars{},
		expected: testEmptyCollectionsPtrScalars{
			PtrString:  aws.String("test string"),
			PtrUint8:   aws.Uint8(1),
			PtrUint16:  aws.Uint16(2),
			PtrUint32:  aws.Uint32(3),
			PtrUint64:  aws.Uint64(4),
			PtrInt8:    aws.Int8(-5),
			PtrInt16:   aws.Int16(-6),
			PtrInt32:   aws.Int32(-7),
			PtrInt64:   aws.Int64(-8),
			PtrFloat32: aws.Float32(9.9),
			PtrFloat64: aws.Float64(10.1),
		},
	},
	// omittable nil pointer scalars
	7: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{},
		},
		actual:   &testEmptyCollectionsOmittedPtrNumericalScalars{},
		expected: testEmptyCollectionsOmittedPtrNumericalScalars{},
	},
	// omittable non-nil pointer to scalars with zero value
	8: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"PtrUint8":   {N: aws.String("0")},
				"PtrUint16":  {N: aws.String("0")},
				"PtrUint32":  {N: aws.String("0")},
				"PtrUint64":  {N: aws.String("0")},
				"PtrInt8":    {N: aws.String("0")},
				"PtrInt16":   {N: aws.String("0")},
				"PtrInt32":   {N: aws.String("0")},
				"PtrInt64":   {N: aws.String("0")},
				"PtrFloat32": {N: aws.String("0")},
				"PtrFloat64": {N: aws.String("0")},
			},
		},
		actual: &testEmptyCollectionsOmittedPtrNumericalScalars{},
		expected: testEmptyCollectionsOmittedPtrNumericalScalars{
			PtrUint8:   aws.Uint8(0),
			PtrUint16:  aws.Uint16(0),
			PtrUint32:  aws.Uint32(0),
			PtrUint64:  aws.Uint64(0),
			PtrInt8:    aws.Int8(0),
			PtrInt16:   aws.Int16(0),
			PtrInt32:   aws.Int32(0),
			PtrInt64:   aws.Int64(0),
			PtrFloat32: aws.Float32(0),
			PtrFloat64: aws.Float64(0),
		},
	},
	// omittable non-nil pointer to non-zero scalar
	9: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"PtrUint8":   {N: aws.String("1")},
				"PtrUint16":  {N: aws.String("2")},
				"PtrUint32":  {N: aws.String("3")},
				"PtrUint64":  {N: aws.String("4")},
				"PtrInt8":    {N: aws.String("-5")},
				"PtrInt16":   {N: aws.String("-6")},
				"PtrInt32":   {N: aws.String("-7")},
				"PtrInt64":   {N: aws.String("-8")},
				"PtrFloat32": {N: aws.String("9.9")},
				"PtrFloat64": {N: aws.String("10.1")},
			},
		},
		actual: &testEmptyCollectionsOmittedPtrNumericalScalars{},
		expected: testEmptyCollectionsOmittedPtrNumericalScalars{
			PtrUint8:   aws.Uint8(1),
			PtrUint16:  aws.Uint16(2),
			PtrUint32:  aws.Uint32(3),
			PtrUint64:  aws.Uint64(4),
			PtrInt8:    aws.Int8(-5),
			PtrInt16:   aws.Int16(-6),
			PtrInt32:   aws.Int32(-7),
			PtrInt64:   aws.Int64(-8),
			PtrFloat32: aws.Float32(9.9),
			PtrFloat64: aws.Float64(10.1),
		},
	},
	// maps, slices nil values
	10: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Map":       {NULL: aws.Bool(true)},
				"Slice":     {NULL: aws.Bool(true)},
				"ByteSlice": {NULL: aws.Bool(true)},
				"ByteArray": {B: make([]byte, 4)},
				"ZeroArray": {B: make([]byte, 0)},
				"BinarySet": {NULL: aws.Bool(true)},
				"NumberSet": {NULL: aws.Bool(true)},
				"StringSet": {NULL: aws.Bool(true)},
			},
		},
		actual:   &testEmptyCollectionTypes{},
		expected: testEmptyCollectionTypes{},
	},
	// maps, slices zero values
	11: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Map":       {M: map[string]*dynamodb.AttributeValue{}},
				"Slice":     {L: []*dynamodb.AttributeValue{}},
				"ByteSlice": {B: []byte{}},
				"ByteArray": {B: make([]byte, 4)},
				"ZeroArray": {B: make([]byte, 0)},
				"BinarySet": {BS: [][]byte{}},
				"NumberSet": {NS: []*string{}},
				"StringSet": {SS: []*string{}},
			},
		},
		actual: &testEmptyCollectionTypes{},
		expected: testEmptyCollectionTypes{
			Map:       map[string]string{},
			Slice:     []string{},
			ByteSlice: []byte{},
			ByteArray: [4]byte{},
			ZeroArray: [0]byte{},
			BinarySet: [][]byte{},
			NumberSet: []int{},
			StringSet: []string{},
		},
	},
	// maps, slices non-zero values
	12: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Map": {
					M: map[string]*dynamodb.AttributeValue{
						"key": {S: aws.String("value")},
					},
				},
				"Slice":     {L: []*dynamodb.AttributeValue{{S: aws.String("test")}, {S: aws.String("slice")}}},
				"ByteSlice": {B: []byte{0, 1}},
				"ByteArray": {B: []byte{0, 1, 2, 3}},
				"ZeroArray": {B: make([]byte, 0)},
				"BinarySet": {BS: [][]byte{{0, 1}, {2, 3}}},
				"NumberSet": {NS: []*string{aws.String("0"), aws.String("1")}},
				"StringSet": {SS: []*string{aws.String("test"), aws.String("slice")}},
			},
		},
		actual: &testEmptyCollectionTypes{},
		expected: testEmptyCollectionTypes{
			Map:       map[string]string{"key": "value"},
			Slice:     []string{"test", "slice"},
			ByteSlice: []byte{0, 1},
			ByteArray: [4]byte{0, 1, 2, 3},
			ZeroArray: [0]byte{},
			BinarySet: [][]byte{{0, 1}, {2, 3}},
			NumberSet: []int{0, 1},
			StringSet: []string{"test", "slice"},
		},
	},
	// omittable maps, slices nil values
	13: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"ByteArray": {B: make([]byte, 4)},
			},
		},
		actual:   &testEmptyCollectionTypesOmitted{},
		expected: testEmptyCollectionTypesOmitted{},
	},
	// omittable maps, slices zero values
	14: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Map":       {M: map[string]*dynamodb.AttributeValue{}},
				"Slice":     {L: []*dynamodb.AttributeValue{}},
				"ByteSlice": {B: []byte{}},
				"ByteArray": {B: make([]byte, 4)},
				"BinarySet": {BS: [][]byte{}},
				"NumberSet": {NS: []*string{}},
				"StringSet": {SS: []*string{}},
			},
		},
		actual: &testEmptyCollectionTypesOmitted{},
		expected: testEmptyCollectionTypesOmitted{
			Map:       map[string]string{},
			Slice:     []string{},
			ByteSlice: []byte{},
			ByteArray: [4]byte{},
			BinarySet: [][]byte{},
			NumberSet: []int{},
			StringSet: []string{},
		},
	},
	// omittable maps, slices non-zero values
	15: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Map": {
					M: map[string]*dynamodb.AttributeValue{
						"key": {S: aws.String("value")},
					},
				},
				"Slice":     {L: []*dynamodb.AttributeValue{{S: aws.String("test")}, {S: aws.String("slice")}}},
				"ByteSlice": {B: []byte{0, 1}},
				"ByteArray": {B: []byte{0, 1, 2, 3}},
				"BinarySet": {BS: [][]byte{{0, 1}, {2, 3}}},
				"NumberSet": {NS: []*string{aws.String("0"), aws.String("1")}},
				"StringSet": {SS: []*string{aws.String("test"), aws.String("slice")}},
			},
		},
		actual: &testEmptyCollectionTypesOmitted{},
		expected: testEmptyCollectionTypesOmitted{
			Map:       map[string]string{"key": "value"},
			Slice:     []string{"test", "slice"},
			ByteSlice: []byte{0, 1},
			ByteArray: [4]byte{0, 1, 2, 3},
			ZeroArray: [0]byte{},
			BinarySet: [][]byte{{0, 1}, {2, 3}},
			NumberSet: []int{0, 1},
			StringSet: []string{"test", "slice"},
		},
	},
	// structs with members zero
	16: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Struct": {
					M: map[string]*dynamodb.AttributeValue{
						"Int": {N: aws.String("0")},
					},
				},
				"PtrStruct": {NULL: aws.Bool(true)},
			},
		},
		actual: &struct {
			Struct    testEmptyCollectionStruct
			PtrStruct *testEmptyCollectionStruct
		}{},
		expected: struct {
			Struct    testEmptyCollectionStruct
			PtrStruct *testEmptyCollectionStruct
		}{},
	},
	// structs with members non-zero value
	17: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Struct": {
					M: map[string]*dynamodb.AttributeValue{
						"Int": {N: aws.String("1")},
					},
				},
				"PtrStruct": {
					M: map[string]*dynamodb.AttributeValue{
						"Int": {N: aws.String("1")},
					},
				},
			},
		},
		actual: &struct {
			Struct    testEmptyCollectionStruct
			PtrStruct *testEmptyCollectionStruct
		}{},
		expected: struct {
			Struct    testEmptyCollectionStruct
			PtrStruct *testEmptyCollectionStruct
		}{
			Struct:    testEmptyCollectionStruct{Int: 1},
			PtrStruct: &testEmptyCollectionStruct{Int: 1},
		},
	},
	// struct with omittable members zero value
	18: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Struct":    {M: map[string]*dynamodb.AttributeValue{}},
				"PtrStruct": {NULL: aws.Bool(true)},
			},
		},
		actual: &struct {
			Struct    testEmptyCollectionStructOmitted
			PtrStruct *testEmptyCollectionStructOmitted
		}{},
		expected: struct {
			Struct    testEmptyCollectionStructOmitted
			PtrStruct *testEmptyCollectionStructOmitted
		}{},
	},
	// omittable struct with omittable members zero value
	19: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Struct": {M: map[string]*dynamodb.AttributeValue{}},
			},
		},
		actual: &struct {
			Struct    testEmptyCollectionStructOmitted  `dynamodbav:",omitempty"`
			PtrStruct *testEmptyCollectionStructOmitted `dynamodbav:",omitempty"`
		}{},
		expected: struct {
			Struct    testEmptyCollectionStructOmitted  `dynamodbav:",omitempty"`
			PtrStruct *testEmptyCollectionStructOmitted `dynamodbav:",omitempty"`
		}{},
	},
	// omittable struct with omittable members non-zero value
	20: {
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Struct": {
					M: map[string]*dynamodb.AttributeValue{
						"Slice": {L: []*dynamodb.AttributeValue{{S: aws.String("test")}}},
					},
				},
				"InitPtrStruct": {
					M: map[string]*dynamodb.AttributeValue{
						"Slice": {L: []*dynamodb.AttributeValue{{S: aws.String("test")}}},
					},
				},
			},
		},
		actual: &struct {
			Struct        testEmptyCollectionStructOmitted  `dynamodbav:",omitempty"`
			InitPtrStruct *testEmptyCollectionStructOmitted `dynamodbav:",omitempty"`
		}{},
		expected: struct {
			Struct        testEmptyCollectionStructOmitted  `dynamodbav:",omitempty"`
			InitPtrStruct *testEmptyCollectionStructOmitted `dynamodbav:",omitempty"`
		}{
			Struct:        testEmptyCollectionStructOmitted{Slice: []string{"test"}},
			InitPtrStruct: &testEmptyCollectionStructOmitted{Slice: []string{"test"}},
		},
	},
	21: { // empty slice and NullEmptyByteSlice disabled
		encoderOpts: func(encoder *Encoder) {
			encoder.NullEmptyByteSlice = false
		},
		in: &dynamodb.AttributeValue{
			B: []byte{},
		},
		actual:   &[]byte{},
		expected: []byte{},
	},
	22: { // empty slice and NullEmptyByteSlice disabled, and omitempty
		encoderOpts: func(encoder *Encoder) {
			encoder.NullEmptyByteSlice = false
		},
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{},
		},
		actual: &struct {
			Value []byte `dynamodbav:",omitempty"`
		}{},
		expected: struct {
			Value []byte `dynamodbav:",omitempty"`
		}{},
	},
}

func TestMarshalEmptyCollections(t *testing.T) {
	for i, c := range sharedEmptyCollectionsTestCases {
		encoder := NewEncoder(func(e *Encoder) {
			e.EnableEmptyCollections = true
			if c.encoderOpts != nil {
				c.encoderOpts(e)
			}
		})
		av, err := encoder.Encode(c.expected)
		assertConvertTest(t, i, av, c.in, err, c.err)
	}
}

func TestEmptyCollectionsSpecialCases(t *testing.T) {
	// ptr string non nil with empty value

	type SpecialCases struct {
		PtrString        *string
		OmittedPtrString *string `dynamodbav:",omitempty"`
	}

	expectedEncode := &dynamodb.AttributeValue{
		M: map[string]*dynamodb.AttributeValue{
			"PtrString": {NULL: aws.Bool(true)},
		},
	}
	expectedDecode := SpecialCases{}

	encoder := NewEncoder(func(encoder *Encoder) {
		encoder.EnableEmptyCollections = true
	})

	actualEncode, err := encoder.Encode(&SpecialCases{
		PtrString:        aws.String(""),
		OmittedPtrString: aws.String(""),
	})
	if err != nil {
		t.Fatalf("expected no err got %v", err)
	}
	if e, a := expectedEncode, actualEncode; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}

	decoder := NewDecoder(func(decoder *Decoder) {
		decoder.EnableEmptyCollections = true
	})

	var actualDecode SpecialCases
	err = decoder.Decode(&dynamodb.AttributeValue{}, &actualDecode)
	if err != nil {
		t.Fatalf("expected no err got %v", err)
	}
	if e, a := expectedDecode, actualDecode; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestUnmarshalEmptyCollections(t *testing.T) {
	for i, c := range sharedEmptyCollectionsTestCases {
		decoder := NewDecoder(func(d *Decoder) {
			d.EnableEmptyCollections = true
		})
		err := decoder.Decode(c.in, c.actual)
		assertConvertTest(t, i, c.actual, c.expected, err, c.err)
	}
}
