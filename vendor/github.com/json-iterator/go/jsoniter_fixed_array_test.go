package jsoniter

import (
	"encoding/json"
	"github.com/stretchr/testify/require"
	"testing"
)

func Test_encode_fixed_array(t *testing.T) {
	should := require.New(t)
	type FixedArray [2]float64
	fixed := FixedArray{0.1, 1.0}
	output, err := MarshalToString(fixed)
	should.Nil(err)
	should.Equal("[0.1,1]", output)
}

func Test_encode_fixed_array_of_map(t *testing.T) {
	should := require.New(t)
	type FixedArray [2]map[string]string
	fixed := FixedArray{map[string]string{"1": "2"}, map[string]string{"3": "4"}}
	output, err := MarshalToString(fixed)
	should.Nil(err)
	should.Equal(`[{"1":"2"},{"3":"4"}]`, output)
}

func Test_decode_fixed_array(t *testing.T) {
	should := require.New(t)
	type FixedArray [2]float64
	var fixed FixedArray
	should.Nil(json.Unmarshal([]byte("[1,2,3]"), &fixed))
	should.Equal(float64(1), fixed[0])
	should.Equal(float64(2), fixed[1])
	should.Nil(Unmarshal([]byte("[1,2,3]"), &fixed))
	should.Equal(float64(1), fixed[0])
	should.Equal(float64(2), fixed[1])
}
