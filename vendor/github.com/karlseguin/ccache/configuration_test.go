package ccache

import (
	. "github.com/karlseguin/expect"
	"testing"
)

type ConfigurationTests struct{}

func Test_Configuration(t *testing.T) {
	Expectify(new(ConfigurationTests), t)
}

func (_ *ConfigurationTests) BucketsPowerOf2() {
	for i := uint32(0); i < 31; i++ {
		c := Configure().Buckets(i)
		if i == 1 || i == 2 || i == 4 || i == 8 || i == 16 {
			Expect(c.buckets).ToEqual(int(i))
		} else {
			Expect(c.buckets).ToEqual(16)
		}
	}
}
