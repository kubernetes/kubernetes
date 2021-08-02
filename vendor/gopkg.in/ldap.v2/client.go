package ldap

import (
	"crypto/tls"
	"time"
)

// Client knows how to interact with an LDAP server
type Client interface {
	Start()
	StartTLS(config *tls.Config) error
	Close()
	SetTimeout(time.Duration)

	Bind(username, password string) error
	SimpleBind(simpleBindRequest *SimpleBindRequest) (*SimpleBindResult, error)

	Add(addRequest *AddRequest) error
	Del(delRequest *DelRequest) error
	Modify(modifyRequest *ModifyRequest) error

	Compare(dn, attribute, value string) (bool, error)
	PasswordModify(passwordModifyRequest *PasswordModifyRequest) (*PasswordModifyResult, error)

	Search(searchRequest *SearchRequest) (*SearchResult, error)
	SearchWithPaging(searchRequest *SearchRequest, pagingSize uint32) (*SearchResult, error)
}
