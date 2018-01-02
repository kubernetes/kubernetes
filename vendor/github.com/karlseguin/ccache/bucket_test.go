package ccache

import (
	. "github.com/karlseguin/expect"
	"testing"
	"time"
)

type BucketTests struct {
}

func Tests_Bucket(t *testing.T) {
	Expectify(new(BucketTests), t)
}

func (_ *BucketTests) GetMissFromBucket() {
	bucket := testBucket()
	Expect(bucket.get("invalid")).To.Equal(nil)
}

func (_ *BucketTests) GetHitFromBucket() {
	bucket := testBucket()
	item := bucket.get("power")
	assertValue(item, "9000")
}

func (_ *BucketTests) DeleteItemFromBucket() {
	bucket := testBucket()
	bucket.delete("power")
	Expect(bucket.get("power")).To.Equal(nil)
}

func (_ *BucketTests) SetsANewBucketItem() {
	bucket := testBucket()
	item, existing := bucket.set("spice", TestValue("flow"), time.Minute)
	assertValue(item, "flow")
	item = bucket.get("spice")
	assertValue(item, "flow")
	Expect(existing).To.Equal(nil)
}

func (_ *BucketTests) SetsAnExistingItem() {
	bucket := testBucket()
	item, existing := bucket.set("power", TestValue("9001"), time.Minute)
	assertValue(item, "9002")
	item = bucket.get("power")
	assertValue(item, "9002")
	assertValue(existing, "9001")
}

func testBucket() *bucket {
	b := &bucket{lookup: make(map[string]*Item)}
	b.lookup["power"] = &Item{
		key:   "power",
		value: TestValue("9000"),
	}
	return b
}

func assertValue(item *Item, expected string) {
	value := item.value.(TestValue)
	Expect(value).To.Equal(TestValue(expected))
}

type TestValue string

func (v TestValue) Expires() time.Time {
	return time.Now()
}
