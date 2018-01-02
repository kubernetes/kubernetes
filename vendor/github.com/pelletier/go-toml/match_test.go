package toml

import (
	"fmt"
	"testing"
)

// dump path tree to a string
func pathString(root pathFn) string {
	result := fmt.Sprintf("%T:", root)
	switch fn := root.(type) {
	case *terminatingFn:
		result += "{}"
	case *matchKeyFn:
		result += fmt.Sprintf("{%s}", fn.Name)
		result += pathString(fn.next)
	case *matchIndexFn:
		result += fmt.Sprintf("{%d}", fn.Idx)
		result += pathString(fn.next)
	case *matchSliceFn:
		result += fmt.Sprintf("{%d:%d:%d}",
			fn.Start, fn.End, fn.Step)
		result += pathString(fn.next)
	case *matchAnyFn:
		result += "{}"
		result += pathString(fn.next)
	case *matchUnionFn:
		result += "{["
		for _, v := range fn.Union {
			result += pathString(v) + ", "
		}
		result += "]}"
	case *matchRecursiveFn:
		result += "{}"
		result += pathString(fn.next)
	case *matchFilterFn:
		result += fmt.Sprintf("{%s}", fn.Name)
		result += pathString(fn.next)
	}
	return result
}

func assertPathMatch(t *testing.T, path, ref *Query) bool {
	pathStr := pathString(path.root)
	refStr := pathString(ref.root)
	if pathStr != refStr {
		t.Errorf("paths do not match")
		t.Log("test:", pathStr)
		t.Log("ref: ", refStr)
		return false
	}
	return true
}

func assertPath(t *testing.T, query string, ref *Query) {
	path, _ := parseQuery(lexQuery(query))
	assertPathMatch(t, path, ref)
}

func buildPath(parts ...pathFn) *Query {
	query := newQuery()
	for _, v := range parts {
		query.appendPath(v)
	}
	return query
}

func TestPathRoot(t *testing.T) {
	assertPath(t,
		"$",
		buildPath(
		// empty
		))
}

func TestPathKey(t *testing.T) {
	assertPath(t,
		"$.foo",
		buildPath(
			newMatchKeyFn("foo"),
		))
}

func TestPathBracketKey(t *testing.T) {
	assertPath(t,
		"$[foo]",
		buildPath(
			newMatchKeyFn("foo"),
		))
}

func TestPathBracketStringKey(t *testing.T) {
	assertPath(t,
		"$['foo']",
		buildPath(
			newMatchKeyFn("foo"),
		))
}

func TestPathIndex(t *testing.T) {
	assertPath(t,
		"$[123]",
		buildPath(
			newMatchIndexFn(123),
		))
}

func TestPathSliceStart(t *testing.T) {
	assertPath(t,
		"$[123:]",
		buildPath(
			newMatchSliceFn(123, maxInt, 1),
		))
}

func TestPathSliceStartEnd(t *testing.T) {
	assertPath(t,
		"$[123:456]",
		buildPath(
			newMatchSliceFn(123, 456, 1),
		))
}

func TestPathSliceStartEndColon(t *testing.T) {
	assertPath(t,
		"$[123:456:]",
		buildPath(
			newMatchSliceFn(123, 456, 1),
		))
}

func TestPathSliceStartStep(t *testing.T) {
	assertPath(t,
		"$[123::7]",
		buildPath(
			newMatchSliceFn(123, maxInt, 7),
		))
}

func TestPathSliceEndStep(t *testing.T) {
	assertPath(t,
		"$[:456:7]",
		buildPath(
			newMatchSliceFn(0, 456, 7),
		))
}

func TestPathSliceStep(t *testing.T) {
	assertPath(t,
		"$[::7]",
		buildPath(
			newMatchSliceFn(0, maxInt, 7),
		))
}

func TestPathSliceAll(t *testing.T) {
	assertPath(t,
		"$[123:456:7]",
		buildPath(
			newMatchSliceFn(123, 456, 7),
		))
}

func TestPathAny(t *testing.T) {
	assertPath(t,
		"$.*",
		buildPath(
			newMatchAnyFn(),
		))
}

func TestPathUnion(t *testing.T) {
	assertPath(t,
		"$[foo, bar, baz]",
		buildPath(
			&matchUnionFn{[]pathFn{
				newMatchKeyFn("foo"),
				newMatchKeyFn("bar"),
				newMatchKeyFn("baz"),
			}},
		))
}

func TestPathRecurse(t *testing.T) {
	assertPath(t,
		"$..*",
		buildPath(
			newMatchRecursiveFn(),
		))
}

func TestPathFilterExpr(t *testing.T) {
	assertPath(t,
		"$[?('foo'),?(bar)]",
		buildPath(
			&matchUnionFn{[]pathFn{
				newMatchFilterFn("foo", Position{}),
				newMatchFilterFn("bar", Position{}),
			}},
		))
}
