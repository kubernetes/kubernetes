// +build linux solaris

package libcontainerd

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestSerialization(t *testing.T) {
	var (
		q             queue
		serialization = 1
	)

	q.append("aaa", func() {
		//simulate a long time task
		time.Sleep(10 * time.Millisecond)
		require.EqualValues(t, serialization, 1)
		serialization = 2
	})
	q.append("aaa", func() {
		require.EqualValues(t, serialization, 2)
		serialization = 3
	})
	q.append("aaa", func() {
		require.EqualValues(t, serialization, 3)
		serialization = 4
	})
	time.Sleep(20 * time.Millisecond)
}
