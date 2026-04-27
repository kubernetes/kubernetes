package spdy

// FramerOption allows callers to customize frame parsing limits.
type FramerOption func(*Framer)

// WithMaxControlFramePayloadSize sets the control-frame payload limit.
func WithMaxControlFramePayloadSize(size uint32) FramerOption {
	return func(f *Framer) {
		f.maxFrameLength = size
	}
}

// WithMaxHeaderFieldSize sets the per-header name/value size limit.
func WithMaxHeaderFieldSize(size uint32) FramerOption {
	return func(f *Framer) {
		f.maxHeaderFieldSize = size
	}
}

// WithMaxHeaderCount sets the maximum number of headers in a frame.
func WithMaxHeaderCount(count uint32) FramerOption {
	return func(f *Framer) {
		f.maxHeaderCount = count
	}
}
