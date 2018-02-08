//+build jsoniter_sloppy

package jsoniter

// sloppy but faster implementation, do not validate the input json

func (iter *Iterator) skipNumber() {
	for {
		for i := iter.head; i < iter.tail; i++ {
			c := iter.buf[i]
			switch c {
			case ' ', '\n', '\r', '\t', ',', '}', ']':
				iter.head = i
				return
			}
		}
		if !iter.loadMore() {
			return
		}
	}
}

func (iter *Iterator) skipArray() {
	level := 1
	for {
		for i := iter.head; i < iter.tail; i++ {
			switch iter.buf[i] {
			case '"': // If inside string, skip it
				iter.head = i + 1
				iter.skipString()
				i = iter.head - 1 // it will be i++ soon
			case '[': // If open symbol, increase level
				level++
			case ']': // If close symbol, increase level
				level--

				// If we have returned to the original level, we're done
				if level == 0 {
					iter.head = i + 1
					return
				}
			}
		}
		if !iter.loadMore() {
			iter.ReportError("skipObject", "incomplete array")
			return
		}
	}
}

func (iter *Iterator) skipObject() {
	level := 1
	for {
		for i := iter.head; i < iter.tail; i++ {
			switch iter.buf[i] {
			case '"': // If inside string, skip it
				iter.head = i + 1
				iter.skipString()
				i = iter.head - 1 // it will be i++ soon
			case '{': // If open symbol, increase level
				level++
			case '}': // If close symbol, increase level
				level--

				// If we have returned to the original level, we're done
				if level == 0 {
					iter.head = i + 1
					return
				}
			}
		}
		if !iter.loadMore() {
			iter.ReportError("skipObject", "incomplete object")
			return
		}
	}
}

func (iter *Iterator) skipString() {
	for {
		end, escaped := iter.findStringEnd()
		if end == -1 {
			if !iter.loadMore() {
				iter.ReportError("skipString", "incomplete string")
				return
			}
			if escaped {
				iter.head = 1 // skip the first char as last char read is \
			}
		} else {
			iter.head = end
			return
		}
	}
}

// adapted from: https://github.com/buger/jsonparser/blob/master/parser.go
// Tries to find the end of string
// Support if string contains escaped quote symbols.
func (iter *Iterator) findStringEnd() (int, bool) {
	escaped := false
	for i := iter.head; i < iter.tail; i++ {
		c := iter.buf[i]
		if c == '"' {
			if !escaped {
				return i + 1, false
			}
			j := i - 1
			for {
				if j < iter.head || iter.buf[j] != '\\' {
					// even number of backslashes
					// either end of buffer, or " found
					return i + 1, true
				}
				j--
				if j < iter.head || iter.buf[j] != '\\' {
					// odd number of backslashes
					// it is \" or \\\"
					break
				}
				j--
			}
		} else if c == '\\' {
			escaped = true
		}
	}
	j := iter.tail - 1
	for {
		if j < iter.head || iter.buf[j] != '\\' {
			// even number of backslashes
			// either end of buffer, or " found
			return -1, false // do not end with \
		}
		j--
		if j < iter.head || iter.buf[j] != '\\' {
			// odd number of backslashes
			// it is \" or \\\"
			break
		}
		j--

	}
	return -1, true // end with \
}
