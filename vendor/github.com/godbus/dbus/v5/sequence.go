package dbus

// Sequence represents the value of a monotonically increasing counter.
type Sequence uint64

const (
	// NoSequence indicates the absence of a sequence value.
	NoSequence Sequence = 0
)

// sequenceGenerator represents a monotonically increasing counter.
type sequenceGenerator struct {
	nextSequence Sequence
}

func (generator *sequenceGenerator) next() Sequence {
	result := generator.nextSequence
	generator.nextSequence++
	return result
}

func newSequenceGenerator() *sequenceGenerator {
	return &sequenceGenerator{nextSequence: 1}
}
