package oauthdiscovery

import (
	"path"
	"strings"
)

const (
	AuthorizePath = "/authorize"
	TokenPath     = "/token"
	InfoPath      = "/info"

	RequestTokenEndpoint  = "/token/request"
	DisplayTokenEndpoint  = "/token/display"
	ImplicitTokenEndpoint = "/token/implicit"
)

const OpenShiftOAuthAPIPrefix = "/oauth"

func OpenShiftOAuthAuthorizeURL(masterAddr string) string {
	return openShiftOAuthURL(masterAddr, AuthorizePath)
}
func OpenShiftOAuthTokenURL(masterAddr string) string {
	return openShiftOAuthURL(masterAddr, TokenPath)
}
func OpenShiftOAuthTokenRequestURL(masterAddr string) string {
	return openShiftOAuthURL(masterAddr, RequestTokenEndpoint)
}
func OpenShiftOAuthTokenDisplayURL(masterAddr string) string {
	return openShiftOAuthURL(masterAddr, DisplayTokenEndpoint)
}
func OpenShiftOAuthTokenImplicitURL(masterAddr string) string {
	return openShiftOAuthURL(masterAddr, ImplicitTokenEndpoint)
}
func openShiftOAuthURL(masterAddr, oauthEndpoint string) string {
	return strings.TrimRight(masterAddr, "/") + path.Join(OpenShiftOAuthAPIPrefix, oauthEndpoint)
}
