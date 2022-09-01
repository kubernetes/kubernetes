package ldap

import (
	"crypto/tls"
	"time"
)

// Client knows how to interact with an LDAP server
type Client interface {
	Start()
	StartTLS(*tls.Config) error
	Close()
	IsClosing() bool
	SetTimeout(time.Duration)

	Bind(username, password string) error
	UnauthenticatedBind(username string) error
	SimpleBind(*SimpleBindRequest) (*SimpleBindResult, error)
	ExternalBind() error

	Add(*AddRequest) error
	Del(*DelRequest) error
	Modify(*ModifyRequest) error
	ModifyDN(*ModifyDNRequest) error
	ModifyWithResult(*ModifyRequest) (*ModifyResult, error)

	Compare(dn, attribute, value string) (bool, error)
	PasswordModify(*PasswordModifyRequest) (*PasswordModifyResult, error)

	Search(*SearchRequest) (*SearchResult, error)
	SearchWithPaging(searchRequest *SearchRequest, pagingSize uint32) (*SearchResult, error)
}
