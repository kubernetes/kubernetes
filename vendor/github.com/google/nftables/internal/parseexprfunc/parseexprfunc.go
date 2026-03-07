package parseexprfunc

import (
	"github.com/mdlayher/netlink"
)

var (
	ParseExprBytesFromNameFunc func(fam byte, ad *netlink.AttributeDecoder, exprName string) ([]interface{}, error)
	ParseExprBytesFunc         func(fam byte, ad *netlink.AttributeDecoder) ([]interface{}, error)
	ParseExprMsgFunc           func(fam byte, b []byte) ([]interface{}, error)
)
