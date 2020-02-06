package ssautil

import (
	"honnef.co/go/tools/ssa"
)

func Reachable(from, to *ssa.BasicBlock) bool {
	if from == to {
		return true
	}
	if from.Dominates(to) {
		return true
	}

	found := false
	Walk(from, func(b *ssa.BasicBlock) bool {
		if b == to {
			found = true
			return false
		}
		return true
	})
	return found
}

func Walk(b *ssa.BasicBlock, fn func(*ssa.BasicBlock) bool) {
	seen := map[*ssa.BasicBlock]bool{}
	wl := []*ssa.BasicBlock{b}
	for len(wl) > 0 {
		b := wl[len(wl)-1]
		wl = wl[:len(wl)-1]
		if seen[b] {
			continue
		}
		seen[b] = true
		if !fn(b) {
			continue
		}
		wl = append(wl, b.Succs...)
	}
}

func Vararg(x *ssa.Slice) ([]ssa.Value, bool) {
	var out []ssa.Value
	slice, ok := x.X.(*ssa.Alloc)
	if !ok || slice.Comment != "varargs" {
		return nil, false
	}
	for _, ref := range *slice.Referrers() {
		idx, ok := ref.(*ssa.IndexAddr)
		if !ok {
			continue
		}
		v := (*idx.Referrers())[0].(*ssa.Store).Val
		out = append(out, v)
	}
	return out, true
}
