package mesos

import (
	"bytes"
	"io"
)

type labelList []Label // convenience type, for working with unwrapped Label slices

// Equivalent returns true if left and right have the same labels. Order is not important.
func (left *Labels) Equivalent(right *Labels) bool {
	return labelList(left.GetLabels()).Equivalent(labelList(right.GetLabels()))
}

// Equivalent returns true if left and right have the same labels. Order is not important.
func (left labelList) Equivalent(right labelList) bool {
	if len(left) != len(right) {
		return false
	} else {
		for i := range left {
			found := false
			for j := range right {
				if left[i].Equivalent(right[j]) {
					found = true
					break
				}
			}
			if !found {
				return false
			}
		}
		return true
	}
}

// Equivalent returns true if left and right represent the same Label.
func (left Label) Equivalent(right Label) bool {
	if left.Key != right.Key {
		return false
	}
	if left.Value == nil {
		return right.Value == nil
	} else {
		return right.Value != nil && *left.Value == *right.Value
	}
}

func (left Label) writeTo(w io.Writer) (n int64, err error) {
	write := func(s string) {
		if err != nil {
			return
		}
		var n2 int
		n2, err = io.WriteString(w, s)
		n += int64(n2)
	}
	write(left.Key)
	if s := left.GetValue(); s != "" {
		write("=")
		write(s)
	}
	return
}

func (left *Labels) writeTo(w io.Writer) (n int64, err error) {
	var (
		lab = left.GetLabels()
		n2  int
		n3  int64
	)
	for i := range lab {
		if i > 0 {
			n2, err = io.WriteString(w, ",")
			n += int64(n2)
			if err != nil {
				break
			}
		}
		n3, err = lab[i].writeTo(w)
		n += n3
		if err != nil {
			break
		}
	}
	return
}

func (left *Labels) Format() string {
	if left == nil {
		return ""
	}
	var b bytes.Buffer
	left.writeTo(&b)
	return b.String()
}
