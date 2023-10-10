package apivalidation

import "sync"

type Lazy[T any] struct {
	Calculate func() T
	value     T
	once      sync.Once
}

func NewLazy[T any](calc func() T) Lazy[T] {
	return Lazy[T]{
		Calculate: calc,
	}
}

func (l *Lazy[T]) Value() T {
	l.once.Do(func() {
		l.value = l.Calculate()
	})
	return l.value
}
