package toml

import (
	"fmt"
	"io/ioutil"
	"sort"
	"strings"
	"testing"
	"time"
)

type queryTestNode struct {
	value    interface{}
	position Position
}

func valueString(root interface{}) string {
	result := "" //fmt.Sprintf("%T:", root)
	switch node := root.(type) {
	case *tomlValue:
		return valueString(node.value)
	case *QueryResult:
		items := []string{}
		for i, v := range node.Values() {
			items = append(items, fmt.Sprintf("%s:%s",
				node.Positions()[i].String(), valueString(v)))
		}
		sort.Strings(items)
		result = "[" + strings.Join(items, ", ") + "]"
	case queryTestNode:
		result = fmt.Sprintf("%s:%s",
			node.position.String(), valueString(node.value))
	case []interface{}:
		items := []string{}
		for _, v := range node {
			items = append(items, valueString(v))
		}
		sort.Strings(items)
		result = "[" + strings.Join(items, ", ") + "]"
	case *TomlTree:
		// workaround for unreliable map key ordering
		items := []string{}
		for _, k := range node.Keys() {
			v := node.GetPath([]string{k})
			items = append(items, k+":"+valueString(v))
		}
		sort.Strings(items)
		result = "{" + strings.Join(items, ", ") + "}"
	case map[string]interface{}:
		// workaround for unreliable map key ordering
		items := []string{}
		for k, v := range node {
			items = append(items, k+":"+valueString(v))
		}
		sort.Strings(items)
		result = "{" + strings.Join(items, ", ") + "}"
	case int64:
		result += fmt.Sprintf("%d", node)
	case string:
		result += "'" + node + "'"
	case float64:
		result += fmt.Sprintf("%f", node)
	case bool:
		result += fmt.Sprintf("%t", node)
	case time.Time:
		result += fmt.Sprintf("'%v'", node)
	}
	return result
}

func assertValue(t *testing.T, result, ref interface{}) {
	pathStr := valueString(result)
	refStr := valueString(ref)
	if pathStr != refStr {
		t.Errorf("values do not match")
		t.Log("test:", pathStr)
		t.Log("ref: ", refStr)
	}
}

func assertQueryPositions(t *testing.T, toml, query string, ref []interface{}) {
	tree, err := Load(toml)
	if err != nil {
		t.Errorf("Non-nil toml parse error: %v", err)
		return
	}
	q, err := CompileQuery(query)
	if err != nil {
		t.Error(err)
		return
	}
	results := q.Execute(tree)
	assertValue(t, results, ref)
}

func TestQueryRoot(t *testing.T) {
	assertQueryPositions(t,
		"a = 42",
		"$",
		[]interface{}{
			queryTestNode{
				map[string]interface{}{
					"a": int64(42),
				}, Position{1, 1},
			},
		})
}

func TestQueryKey(t *testing.T) {
	assertQueryPositions(t,
		"[foo]\na = 42",
		"$.foo.a",
		[]interface{}{
			queryTestNode{
				int64(42), Position{2, 1},
			},
		})
}

func TestQueryKeyString(t *testing.T) {
	assertQueryPositions(t,
		"[foo]\na = 42",
		"$.foo['a']",
		[]interface{}{
			queryTestNode{
				int64(42), Position{2, 1},
			},
		})
}

func TestQueryIndex(t *testing.T) {
	assertQueryPositions(t,
		"[foo]\na = [1,2,3,4,5,6,7,8,9,0]",
		"$.foo.a[5]",
		[]interface{}{
			queryTestNode{
				int64(6), Position{2, 1},
			},
		})
}

func TestQuerySliceRange(t *testing.T) {
	assertQueryPositions(t,
		"[foo]\na = [1,2,3,4,5,6,7,8,9,0]",
		"$.foo.a[0:5]",
		[]interface{}{
			queryTestNode{
				int64(1), Position{2, 1},
			},
			queryTestNode{
				int64(2), Position{2, 1},
			},
			queryTestNode{
				int64(3), Position{2, 1},
			},
			queryTestNode{
				int64(4), Position{2, 1},
			},
			queryTestNode{
				int64(5), Position{2, 1},
			},
		})
}

func TestQuerySliceStep(t *testing.T) {
	assertQueryPositions(t,
		"[foo]\na = [1,2,3,4,5,6,7,8,9,0]",
		"$.foo.a[0:5:2]",
		[]interface{}{
			queryTestNode{
				int64(1), Position{2, 1},
			},
			queryTestNode{
				int64(3), Position{2, 1},
			},
			queryTestNode{
				int64(5), Position{2, 1},
			},
		})
}

func TestQueryAny(t *testing.T) {
	assertQueryPositions(t,
		"[foo.bar]\na=1\nb=2\n[foo.baz]\na=3\nb=4",
		"$.foo.*",
		[]interface{}{
			queryTestNode{
				map[string]interface{}{
					"a": int64(1),
					"b": int64(2),
				}, Position{1, 1},
			},
			queryTestNode{
				map[string]interface{}{
					"a": int64(3),
					"b": int64(4),
				}, Position{4, 1},
			},
		})
}
func TestQueryUnionSimple(t *testing.T) {
	assertQueryPositions(t,
		"[foo.bar]\na=1\nb=2\n[baz.foo]\na=3\nb=4\n[gorf.foo]\na=5\nb=6",
		"$.*[bar,foo]",
		[]interface{}{
			queryTestNode{
				map[string]interface{}{
					"a": int64(1),
					"b": int64(2),
				}, Position{1, 1},
			},
			queryTestNode{
				map[string]interface{}{
					"a": int64(3),
					"b": int64(4),
				}, Position{4, 1},
			},
			queryTestNode{
				map[string]interface{}{
					"a": int64(5),
					"b": int64(6),
				}, Position{7, 1},
			},
		})
}

func TestQueryRecursionAll(t *testing.T) {
	assertQueryPositions(t,
		"[foo.bar]\na=1\nb=2\n[baz.foo]\na=3\nb=4\n[gorf.foo]\na=5\nb=6",
		"$..*",
		[]interface{}{
			queryTestNode{
				map[string]interface{}{
					"foo": map[string]interface{}{
						"bar": map[string]interface{}{
							"a": int64(1),
							"b": int64(2),
						},
					},
					"baz": map[string]interface{}{
						"foo": map[string]interface{}{
							"a": int64(3),
							"b": int64(4),
						},
					},
					"gorf": map[string]interface{}{
						"foo": map[string]interface{}{
							"a": int64(5),
							"b": int64(6),
						},
					},
				}, Position{1, 1},
			},
			queryTestNode{
				map[string]interface{}{
					"bar": map[string]interface{}{
						"a": int64(1),
						"b": int64(2),
					},
				}, Position{1, 1},
			},
			queryTestNode{
				map[string]interface{}{
					"a": int64(1),
					"b": int64(2),
				}, Position{1, 1},
			},
			queryTestNode{
				int64(1), Position{2, 1},
			},
			queryTestNode{
				int64(2), Position{3, 1},
			},
			queryTestNode{
				map[string]interface{}{
					"foo": map[string]interface{}{
						"a": int64(3),
						"b": int64(4),
					},
				}, Position{4, 1},
			},
			queryTestNode{
				map[string]interface{}{
					"a": int64(3),
					"b": int64(4),
				}, Position{4, 1},
			},
			queryTestNode{
				int64(3), Position{5, 1},
			},
			queryTestNode{
				int64(4), Position{6, 1},
			},
			queryTestNode{
				map[string]interface{}{
					"foo": map[string]interface{}{
						"a": int64(5),
						"b": int64(6),
					},
				}, Position{7, 1},
			},
			queryTestNode{
				map[string]interface{}{
					"a": int64(5),
					"b": int64(6),
				}, Position{7, 1},
			},
			queryTestNode{
				int64(5), Position{8, 1},
			},
			queryTestNode{
				int64(6), Position{9, 1},
			},
		})
}

func TestQueryRecursionUnionSimple(t *testing.T) {
	assertQueryPositions(t,
		"[foo.bar]\na=1\nb=2\n[baz.foo]\na=3\nb=4\n[gorf.foo]\na=5\nb=6",
		"$..['foo','bar']",
		[]interface{}{
			queryTestNode{
				map[string]interface{}{
					"bar": map[string]interface{}{
						"a": int64(1),
						"b": int64(2),
					},
				}, Position{1, 1},
			},
			queryTestNode{
				map[string]interface{}{
					"a": int64(3),
					"b": int64(4),
				}, Position{4, 1},
			},
			queryTestNode{
				map[string]interface{}{
					"a": int64(1),
					"b": int64(2),
				}, Position{1, 1},
			},
			queryTestNode{
				map[string]interface{}{
					"a": int64(5),
					"b": int64(6),
				}, Position{7, 1},
			},
		})
}

func TestQueryFilterFn(t *testing.T) {
	buff, err := ioutil.ReadFile("example.toml")
	if err != nil {
		t.Error(err)
		return
	}

	assertQueryPositions(t, string(buff),
		"$..[?(int)]",
		[]interface{}{
			queryTestNode{
				int64(8001), Position{13, 1},
			},
			queryTestNode{
				int64(8001), Position{13, 1},
			},
			queryTestNode{
				int64(8002), Position{13, 1},
			},
			queryTestNode{
				int64(5000), Position{14, 1},
			},
		})

	assertQueryPositions(t, string(buff),
		"$..[?(string)]",
		[]interface{}{
			queryTestNode{
				"TOML Example", Position{3, 1},
			},
			queryTestNode{
				"Tom Preston-Werner", Position{6, 1},
			},
			queryTestNode{
				"GitHub", Position{7, 1},
			},
			queryTestNode{
				"GitHub Cofounder & CEO\nLikes tater tots and beer.",
				Position{8, 1},
			},
			queryTestNode{
				"192.168.1.1", Position{12, 1},
			},
			queryTestNode{
				"10.0.0.1", Position{21, 3},
			},
			queryTestNode{
				"eqdc10", Position{22, 3},
			},
			queryTestNode{
				"10.0.0.2", Position{25, 3},
			},
			queryTestNode{
				"eqdc10", Position{26, 3},
			},
		})

	assertQueryPositions(t, string(buff),
		"$..[?(float)]",
		[]interface{}{
		// no float values in document
		})

	tv, _ := time.Parse(time.RFC3339, "1979-05-27T07:32:00Z")
	assertQueryPositions(t, string(buff),
		"$..[?(tree)]",
		[]interface{}{
			queryTestNode{
				map[string]interface{}{
					"name":         "Tom Preston-Werner",
					"organization": "GitHub",
					"bio":          "GitHub Cofounder & CEO\nLikes tater tots and beer.",
					"dob":          tv,
				}, Position{5, 1},
			},
			queryTestNode{
				map[string]interface{}{
					"server":         "192.168.1.1",
					"ports":          []interface{}{int64(8001), int64(8001), int64(8002)},
					"connection_max": int64(5000),
					"enabled":        true,
				}, Position{11, 1},
			},
			queryTestNode{
				map[string]interface{}{
					"alpha": map[string]interface{}{
						"ip": "10.0.0.1",
						"dc": "eqdc10",
					},
					"beta": map[string]interface{}{
						"ip": "10.0.0.2",
						"dc": "eqdc10",
					},
				}, Position{17, 1},
			},
			queryTestNode{
				map[string]interface{}{
					"ip": "10.0.0.1",
					"dc": "eqdc10",
				}, Position{20, 3},
			},
			queryTestNode{
				map[string]interface{}{
					"ip": "10.0.0.2",
					"dc": "eqdc10",
				}, Position{24, 3},
			},
			queryTestNode{
				map[string]interface{}{
					"data": []interface{}{
						[]interface{}{"gamma", "delta"},
						[]interface{}{int64(1), int64(2)},
					},
				}, Position{28, 1},
			},
		})

	assertQueryPositions(t, string(buff),
		"$..[?(time)]",
		[]interface{}{
			queryTestNode{
				tv, Position{9, 1},
			},
		})

	assertQueryPositions(t, string(buff),
		"$..[?(bool)]",
		[]interface{}{
			queryTestNode{
				true, Position{15, 1},
			},
		})
}
