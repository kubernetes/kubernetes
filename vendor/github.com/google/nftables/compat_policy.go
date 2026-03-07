package nftables

import (
	"fmt"

	"github.com/google/nftables/expr"
	"golang.org/x/sys/unix"
)

const nft_RULE_COMPAT_F_INV uint32 = (1 << 1)
const nft_RULE_COMPAT_F_MASK uint32 = nft_RULE_COMPAT_F_INV

// Used by xt match or target like xt_tcpudp to set compat policy between xtables and nftables
// https://elixir.bootlin.com/linux/v5.12/source/net/netfilter/nft_compat.c#L187
type compatPolicy struct {
	Proto uint32
	Flag  uint32
}

var xtMatchCompatMap map[string]*compatPolicy = map[string]*compatPolicy{
	"tcp": {
		Proto: unix.IPPROTO_TCP,
	},
	"udp": {
		Proto: unix.IPPROTO_UDP,
	},
	"udplite": {
		Proto: unix.IPPROTO_UDPLITE,
	},
	"tcpmss": {
		Proto: unix.IPPROTO_TCP,
	},
	"sctp": {
		Proto: unix.IPPROTO_SCTP,
	},
	"osf": {
		Proto: unix.IPPROTO_TCP,
	},
	"ipcomp": {
		Proto: unix.IPPROTO_COMP,
	},
	"esp": {
		Proto: unix.IPPROTO_ESP,
	},
}

var xtTargetCompatMap map[string]*compatPolicy = map[string]*compatPolicy{
	"TCPOPTSTRIP": {
		Proto: unix.IPPROTO_TCP,
	},
	"TCPMSS": {
		Proto: unix.IPPROTO_TCP,
	},
}

func getCompatPolicy(exprs []expr.Any) (*compatPolicy, error) {
	var exprItem expr.Any
	var compat *compatPolicy

	for _, iter := range exprs {
		var tmpExprItem expr.Any
		var tmpCompat *compatPolicy
		switch item := iter.(type) {
		case *expr.Match:
			if compat, ok := xtMatchCompatMap[item.Name]; ok {
				tmpCompat = compat
				tmpExprItem = item
			} else {
				continue
			}
		case *expr.Target:
			if compat, ok := xtTargetCompatMap[item.Name]; ok {
				tmpCompat = compat
				tmpExprItem = item
			} else {
				continue
			}
		default:
			continue
		}
		if compat == nil {
			compat = tmpCompat
			exprItem = tmpExprItem
		} else if *compat != *tmpCompat {
			return nil, fmt.Errorf("%#v and %#v has conflict compat policy %#v vs %#v", exprItem, tmpExprItem, compat, tmpCompat)
		}
	}
	return compat, nil
}
