//go:build gofuzz
// +build gofuzz

package netlink

import "github.com/google/go-cmp/cmp"

func fuzz(b1 []byte) int {
	// 1. unmarshal, marshal, unmarshal again to check m1 and m2 for equality
	// after a round trip. checkMessage is also used because there is a fair
	// amount of tricky logic around testing for presence of error headers and
	// extended acknowledgement attributes.
	var m1 Message
	if err := m1.UnmarshalBinary(b1); err != nil {
		return 0
	}

	if err := checkMessage(m1); err != nil {
		return 0
	}

	b2, err := m1.MarshalBinary()
	if err != nil {
		panicf("failed to marshal m1: %v", err)
	}

	var m2 Message
	if err := m2.UnmarshalBinary(b2); err != nil {
		panicf("failed to unmarshal m2: %v", err)
	}

	if err := checkMessage(m2); err != nil {
		panicf("failed to check m2: %v", err)
	}

	if diff := cmp.Diff(m1, m2); diff != "" {
		panicf("unexpected Message (-want +got):\n%s", diff)
	}

	// 2. marshal again and compare b2 and b3 (b1 may have reserved bytes set
	// which we ignore and fill with zeros when marshaling) for equality.
	b3, err := m2.MarshalBinary()
	if err != nil {
		panicf("failed to marshal m2: %v", err)
	}

	if diff := cmp.Diff(b2, b3); diff != "" {
		panicf("unexpected message bytes (-want +got):\n%s", diff)
	}

	// 3. unmarshal any possible attributes from m1's data and marshal them
	// again for comparison.
	a1, err := UnmarshalAttributes(m1.Data)
	if err != nil {
		return 0
	}

	ab1, err := MarshalAttributes(a1)
	if err != nil {
		panicf("failed to marshal a1: %v", err)
	}

	a2, err := UnmarshalAttributes(ab1)
	if err != nil {
		panicf("failed to unmarshal a2: %v", err)
	}

	if diff := cmp.Diff(a1, a2); diff != "" {
		panicf("unexpected Attributes (-want +got):\n%s", diff)
	}

	ab2, err := MarshalAttributes(a2)
	if err != nil {
		panicf("failed to marshal a2: %v", err)
	}

	if diff := cmp.Diff(ab1, ab2); diff != "" {
		panicf("unexpected attribute bytes (-want +got):\n%s", diff)
	}

	return 1
}
