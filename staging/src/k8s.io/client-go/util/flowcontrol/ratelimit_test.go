package flowcontrol

import "testing"

func TestLimiter(t *testing.T) {
	valid := []struct {
		rate     float64
		capacity int64
	}{
		{10, 100},
	}

	for _, tt := range valid {
		if _, err := NewBucketLimiter(tt.rate, tt.capacity); err != nil {
			t.Fatalf("%v,%v: %v", tt.rate, tt.capacity, err)
		}
	}
}

func TestLimiterTakeAvailable(t *testing.T) {
	bl, err := NewBucketLimiter(1, 3)
	if err != nil {
		t.Fatalf("unable to create rate limiter")
	}

	t.Logf("available: %v", bl.Available())

	for i := 0; i < 3; i++ {
		if val := bl.TakeAvailable(1); val != 1 {
			t.Errorf("unable to take %v/3: %v", i+1, val)
		}
	}

	if val := bl.TakeAvailable(1); val != 0 {
		t.Errorf("invalid take: %v", val)
	}

}
