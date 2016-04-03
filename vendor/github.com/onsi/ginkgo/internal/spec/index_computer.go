package spec

func ParallelizedIndexRange(length int, parallelTotal int, parallelNode int) (startIndex int, count int) {
	if length == 0 {
		return 0, 0
	}

	// We have more nodes than tests. Trivial case.
	if parallelTotal >= length {
		if parallelNode > length {
			return 0, 0
		} else {
			return parallelNode - 1, 1
		}
	}

	// This is the minimum amount of tests that a node will be required to run
	minTestsPerNode := length / parallelTotal

	// This is the maximum amount of tests that a node will be required to run
	// The algorithm guarantees that this would be equal to at least the minimum amount
	// and at most one more
	maxTestsPerNode := minTestsPerNode
	if length%parallelTotal != 0 {
		maxTestsPerNode++
	}

	// Number of nodes that will have to run the maximum amount of tests per node
	numMaxLoadNodes := length % parallelTotal

	// Number of nodes that precede the current node and will have to run the maximum amount of tests per node
	var numPrecedingMaxLoadNodes int
	if parallelNode > numMaxLoadNodes {
		numPrecedingMaxLoadNodes = numMaxLoadNodes
	} else {
		numPrecedingMaxLoadNodes = parallelNode - 1
	}

	// Number of nodes that precede the current node and will have to run the minimum amount of tests per node
	var numPrecedingMinLoadNodes int
	if parallelNode <= numMaxLoadNodes {
		numPrecedingMinLoadNodes = 0
	} else {
		numPrecedingMinLoadNodes = parallelNode - numMaxLoadNodes - 1
	}

	// Evaluate the test start index and number of tests to run
	startIndex = numPrecedingMaxLoadNodes*maxTestsPerNode + numPrecedingMinLoadNodes*minTestsPerNode
	if parallelNode > numMaxLoadNodes {
		count = minTestsPerNode
	} else {
		count = maxTestsPerNode
	}
	return
}
