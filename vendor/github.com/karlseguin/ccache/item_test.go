package ccache

import (
	"math"
	"testing"
	"time"

	. "github.com/karlseguin/expect"
)

type ItemTests struct{}

func Test_Item(t *testing.T) {
	Expectify(new(ItemTests), t)
}

func (_ *ItemTests) Promotability() {
	item := &Item{promotions: 4}
	Expect(item.shouldPromote(5)).To.Equal(true)
	Expect(item.shouldPromote(5)).To.Equal(false)
}

func (_ *ItemTests) Expired() {
	now := time.Now().UnixNano()
	item1 := &Item{expires: now + (10 * int64(time.Millisecond))}
	item2 := &Item{expires: now - (10 * int64(time.Millisecond))}
	Expect(item1.Expired()).To.Equal(false)
	Expect(item2.Expired()).To.Equal(true)
}

func (_ *ItemTests) TTL() {
	now := time.Now().UnixNano()
	item1 := &Item{expires: now + int64(time.Second)}
	item2 := &Item{expires: now - int64(time.Second)}
	Expect(int(math.Ceil(item1.TTL().Seconds()))).To.Equal(1)
	Expect(int(math.Ceil(item2.TTL().Seconds()))).To.Equal(-1)
}

func (_ *ItemTests) Expires() {
	now := time.Now().UnixNano()
	item := &Item{expires: now + (10)}
	Expect(item.Expires().UnixNano()).To.Equal(now + 10)
}

func (_ *ItemTests) Extend() {
	item := &Item{expires: time.Now().UnixNano() + 10}
	item.Extend(time.Minute * 2)
	Expect(item.Expires().Unix()).To.Equal(time.Now().Unix() + 120)
}
