package internal

import (
	"sync"
)

type memoizedFunc[T any] struct {
	once   sync.Once
	fn     func() (T, error)
	result T
	err    error
}

func (mf *memoizedFunc[T]) do() (T, error) {
	mf.once.Do(func() {
		mf.result, mf.err = mf.fn()
	})
	return mf.result, mf.err
}

// Memoize the result of a function call.
//
// fn is only ever called once, even if it returns an error.
func Memoize[T any](fn func() (T, error)) func() (T, error) {
	return (&memoizedFunc[T]{fn: fn}).do
}
