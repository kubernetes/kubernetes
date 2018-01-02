package jsoniter

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func Test_encode_unexported_field(t *testing.T) {
	type TestData struct {
		a int
		b <-chan int
		C int
		d *time.Timer
	}

	should := require.New(t)

	testChan := make(<-chan int, 10)
	testTimer := time.NewTimer(10 * time.Second)

	obj := &TestData{
		a: 42,
		b: testChan,
		C: 21,
		d: testTimer,
	}

	jb, err := json.Marshal(obj)
	should.NoError(err)
	should.Equal([]byte(`{"C":21}`), jb)

	err = json.Unmarshal([]byte(`{"a": 444, "b":"bad", "C":55, "d":{"not": "a timer"}}`), obj)
	should.NoError(err)
	should.Equal(42, obj.a)
	should.Equal(testChan, obj.b)
	should.Equal(55, obj.C)
	should.Equal(testTimer, obj.d)

	jb, err = Marshal(obj)
	should.NoError(err)
	should.Equal(jb, []byte(`{"C":55}`))

	err = Unmarshal([]byte(`{"a": 444, "b":"bad", "C":256, "d":{"not":"a timer"}}`), obj)
	should.NoError(err)
	should.Equal(42, obj.a)
	should.Equal(testChan, obj.b)
	should.Equal(256, obj.C)
	should.Equal(testTimer, obj.d)
}
