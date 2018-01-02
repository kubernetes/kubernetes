package objx

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestHashWithKey(t *testing.T) {

	assert.Equal(t, "0ce84d8d01f2c7b6e0882b784429c54d280ea2d9", HashWithKey("abc", "def"))

}
