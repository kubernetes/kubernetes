package internal

import "errors"

// DiscardZeroes makes sure that all written bytes are zero
// before discarding them.
type DiscardZeroes struct{}

func (DiscardZeroes) Write(p []byte) (int, error) {
	for _, b := range p {
		if b != 0 {
			return 0, errors.New("encountered non-zero byte")
		}
	}
	return len(p), nil
}
