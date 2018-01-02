// +build linux

package netlink

import (
	"net"
	"testing"

	"golang.org/x/sys/unix"
)

func TestRuleAddDel(t *testing.T) {
	skipUnlessRoot(t)
	defer setUpNetlinkTest(t)()

	srcNet := &net.IPNet{IP: net.IPv4(172, 16, 0, 1), Mask: net.CIDRMask(16, 32)}
	dstNet := &net.IPNet{IP: net.IPv4(172, 16, 1, 1), Mask: net.CIDRMask(24, 32)}

	rulesBegin, err := RuleList(unix.AF_INET)
	if err != nil {
		t.Fatal(err)
	}

	rule := NewRule()
	rule.Table = unix.RT_TABLE_MAIN
	rule.Src = srcNet
	rule.Dst = dstNet
	rule.Priority = 5
	rule.OifName = "lo"
	rule.IifName = "lo"
	rule.Invert = true
	if err := RuleAdd(rule); err != nil {
		t.Fatal(err)
	}

	rules, err := RuleList(unix.AF_INET)
	if err != nil {
		t.Fatal(err)
	}

	if len(rules) != len(rulesBegin)+1 {
		t.Fatal("Rule not added properly")
	}

	// find this rule
	var found bool
	for i := range rules {
		if rules[i].Table == rule.Table &&
			rules[i].Src != nil && rules[i].Src.String() == srcNet.String() &&
			rules[i].Dst != nil && rules[i].Dst.String() == dstNet.String() &&
			rules[i].OifName == rule.OifName &&
			rules[i].Priority == rule.Priority &&
			rules[i].IifName == rule.IifName &&
			rules[i].Invert == rule.Invert {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("Rule has diffrent options than one added")
	}

	if err := RuleDel(rule); err != nil {
		t.Fatal(err)
	}

	rulesEnd, err := RuleList(unix.AF_INET)
	if err != nil {
		t.Fatal(err)
	}

	if len(rulesEnd) != len(rulesBegin) {
		t.Fatal("Rule not removed properly")
	}
}
