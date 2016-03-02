package oidc

type LoginFunc func(ident Identity, sessionKey string) (redirectURL string, err error)
