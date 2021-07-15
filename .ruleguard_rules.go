package gorules

import "github.com/quasilyte/go-ruleguard/dsl"

func netParseIP(m dsl.Matcher) {
	m.Match(`net.ParseIP($_)`).Report("prefer utilnet.ParseIPSloppy()")
}
