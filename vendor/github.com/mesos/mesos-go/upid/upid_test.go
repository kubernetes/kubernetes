package upid

import (
	"testing"
	"testing/quick"

	"github.com/stretchr/testify/assert"
)

func TestUPIDParse(t *testing.T) {
	u, err := Parse("mesos@foo:bar")
	assert.Nil(t, u)
	assert.Error(t, err)

	u, err = Parse("mesoslocalhost5050")
	assert.Nil(t, u)
	assert.Error(t, err)

	u, err = Parse("mesos@localhost")
	assert.Nil(t, u)
	assert.Error(t, err)

	assert.Nil(t, quick.Check(func(s string) bool {
		u, err := Parse(s)
		return u == nil && err != nil
	}, &quick.Config{MaxCount: 100000}))
}

func TestUPIDString(t *testing.T) {
	u, err := Parse("mesos@localhost:5050")
	assert.NotNil(t, u)
	assert.NoError(t, err)
	assert.Equal(t, "mesos@localhost:5050", u.String())
}

func TestUPIDEqual(t *testing.T) {
	u1, err := Parse("mesos@localhost:5050")
	u2, err := Parse("mesos@localhost:5050")
	u3, err := Parse("mesos1@localhost:5050")
	u4, err := Parse("mesos@mesos.com:5050")
	u5, err := Parse("mesos@localhost:5051")
	assert.NoError(t, err)

	assert.True(t, u1.Equal(u2))
	assert.False(t, u1.Equal(u3))
	assert.False(t, u1.Equal(u4))
	assert.False(t, u1.Equal(u5))
	assert.False(t, u1.Equal(nil))
	assert.False(t, (*UPID)(nil).Equal(u5))
	assert.True(t, (*UPID)(nil).Equal(nil))
}
