package blkiodev

import "fmt"

// WeightDevice is a structure that holds device:weight pair
type WeightDevice struct {
	Path   string
	Weight uint16
}

func (w *WeightDevice) String() string {
	return fmt.Sprintf("%s:%d", w.Path, w.Weight)
}

// ThrottleDevice is a structure that holds device:rate_per_second pair
type ThrottleDevice struct {
	Path string
	Rate uint64
}

func (t *ThrottleDevice) String() string {
	return fmt.Sprintf("%s:%d", t.Path, t.Rate)
}
