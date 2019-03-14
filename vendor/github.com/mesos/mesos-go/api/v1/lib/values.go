package mesos

func (left *Value_Scalar) Compare(right *Value_Scalar) int {
	var (
		a = convertToFixed64(left.GetValue())
		b = convertToFixed64(right.GetValue())
	)
	if a < b {
		return -1
	}
	if a > b {
		return 1
	}
	return 0
}

func (left *Value_Ranges) Compare(right *Value_Ranges) int {
	return Ranges(left.GetRange()).Compare(right.GetRange())
}

func (left *Value_Set) Compare(right *Value_Set) int {
	i, j := left.GetItem(), right.GetItem()
	if len(i) <= len(j) {
		b := make(map[string]struct{}, len(j))
		for _, x := range j {
			b[x] = struct{}{}
		}
		// make sure that each item on the left exists on the right,
		// otherwise left is not a subset of right.
		a := make(map[string]struct{}, len(i))
		for _, x := range i {
			if _, ok := b[x]; !ok {
				return 1
			}
			a[x] = struct{}{}
		}
		// if every item on the right also exists on the left, then
		// the sets are equal, otherwise left < right
		for x := range b {
			if _, ok := a[x]; !ok {
				return -1
			}
		}
		return 0
	}
	return 1
}

func (left *Value_Set) Add(right *Value_Set) *Value_Set {
	lefty := left.GetItem()
	righty := right.GetItem()
	c := len(lefty) + len(righty)
	if c == 0 {
		return nil
	}
	m := make(map[string]struct{}, c)
	for _, v := range lefty {
		m[v] = struct{}{}
	}
	for _, v := range righty {
		m[v] = struct{}{}
	}
	x := make([]string, 0, len(m))
	for v := range m {
		x = append(x, v)
	}
	return &Value_Set{Item: x}
}

func (left *Value_Set) Subtract(right *Value_Set) *Value_Set {
	// for each item in right, remove it from left
	lefty := left.GetItem()
	righty := right.GetItem()
	if c := len(lefty); c == 0 {
		return nil
	} else if len(righty) == 0 {
		x := make([]string, c)
		copy(x, lefty)
		return &Value_Set{Item: x}
	}

	a := make(map[string]struct{}, len(lefty))
	for _, x := range lefty {
		a[x] = struct{}{}
	}
	for _, x := range righty {
		delete(a, x)
	}
	if len(a) == 0 {
		return nil
	}
	i := 0
	for k := range a {
		lefty[i] = k
		i++
	}
	return &Value_Set{Item: lefty[:len(a)]}
}

func (left *Value_Ranges) Add(right *Value_Ranges) *Value_Ranges {
	a, b := Ranges(left.GetRange()), Ranges(right.GetRange())
	c := len(a) + len(b)
	if c == 0 {
		return nil
	}
	x := make(Ranges, c)
	if len(a) > 0 {
		copy(x, a)
	}
	if len(b) > 0 {
		copy(x[len(a):], b)
	}
	return &Value_Ranges{
		Range: x.Sort().Squash(),
	}
}

func (left *Value_Ranges) Subtract(right *Value_Ranges) *Value_Ranges {
	a, b := Ranges(left.GetRange()), Ranges(right.GetRange())
	if len(a) > 1 {
		x := make(Ranges, len(a))
		copy(x, a)
		a = x.Sort().Squash()
	}
	for _, r := range b {
		a = a.Remove(r)
	}
	if len(a) == 0 {
		return nil
	}
	return &Value_Ranges{Range: a}
}

func (left *Value_Scalar) Add(right *Value_Scalar) *Value_Scalar {
	sum := convertToFixed64(left.GetValue()) + convertToFixed64(right.GetValue())
	return &Value_Scalar{Value: convertToFloat64(sum)}
}

func (left *Value_Scalar) Subtract(right *Value_Scalar) *Value_Scalar {
	diff := convertToFixed64(left.GetValue()) - convertToFixed64(right.GetValue())
	return &Value_Scalar{Value: convertToFloat64(diff)}
}
