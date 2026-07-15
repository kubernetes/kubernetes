package ldap

import (
	"context"
	"crypto/tls"
	"time"
)

// Client knows how to interact with an LDAP server
type Client interface {
	Start()
	StartTLS(*tls.Config) error
	Close() error
	GetLastError() error
	IsClosing() bool
	SetTimeout(time.Duration)
	TLSConnectionState() (tls.ConnectionState, bool)

	Bind(username, password string) error
	UnauthenticatedBind(username string) error
	SimpleBind(*SimpleBindRequest) (*SimpleBindResult, error)
	ExternalBind() error
	NTLMUnauthenticatedBind(domain, username string) error
	Unbind() error

	Add(*AddRequest) error
	Del(*DelRequest) error
	Modify(*ModifyRequest) error
	ModifyDN(*ModifyDNRequest) error
	ModifyWithResult(*ModifyRequest) (*ModifyResult, error)
	Extended(*ExtendedRequest) (*ExtendedResponse, error)

	Compare(dn, attribute, value string) (bool, error)
	PasswordModify(*PasswordModifyRequest) (*PasswordModifyResult, error)

	Search(*SearchRequest) (*SearchResult, error)
	SearchAsync(ctx context.Context, searchRequest *SearchRequest, bufferSize int) Response
	SearchWithPaging(searchRequest *SearchRequest, pagingSize uint32) (*SearchResult, error)
	DirSync(searchRequest *SearchRequest, flags, maxAttrCount int64, cookie []byte) (*SearchResult, error)
	DirSyncAsync(ctx context.Context, searchRequest *SearchRequest, bufferSize int, flags, maxAttrCount int64, cookie []byte) Response
	Syncrepl(ctx context.Context, searchRequest *SearchRequest, bufferSize int, mode ControlSyncRequestMode, cookie []byte, reloadHint bool) Response
}
