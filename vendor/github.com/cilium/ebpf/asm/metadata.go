package asm

// Metadata contains metadata about an instruction.
type Metadata struct {
	head *metaElement
}

type metaElement struct {
	next       *metaElement
	key, value interface{}
}

// Find the element containing key.
//
// Returns nil if there is no such element.
func (m *Metadata) find(key interface{}) *metaElement {
	for e := m.head; e != nil; e = e.next {
		if e.key == key {
			return e
		}
	}
	return nil
}

// Remove an element from the linked list.
//
// Copies as many elements of the list as necessary to remove r, but doesn't
// perform a full copy.
func (m *Metadata) remove(r *metaElement) {
	current := &m.head
	for e := m.head; e != nil; e = e.next {
		if e == r {
			// We've found the element we want to remove.
			*current = e.next

			// No need to copy the tail.
			return
		}

		// There is another element in front of the one we want to remove.
		// We have to copy it to be able to change metaElement.next.
		cpy := &metaElement{key: e.key, value: e.value}
		*current = cpy
		current = &cpy.next
	}
}

// Set a key to a value.
//
// If value is nil, the key is removed. Avoids modifying old metadata by
// copying if necessary.
func (m *Metadata) Set(key, value interface{}) {
	if e := m.find(key); e != nil {
		if e.value == value {
			// Key is present and the value is the same. Nothing to do.
			return
		}

		// Key is present with a different value. Create a copy of the list
		// which doesn't have the element in it.
		m.remove(e)
	}

	// m.head is now a linked list that doesn't contain key.
	if value == nil {
		return
	}

	m.head = &metaElement{key: key, value: value, next: m.head}
}

// Get the value of a key.
//
// Returns nil if no value with the given key is present.
func (m *Metadata) Get(key interface{}) interface{} {
	if e := m.find(key); e != nil {
		return e.value
	}
	return nil
}
