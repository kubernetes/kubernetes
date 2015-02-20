package osincli

type ClientConfig struct {
	ClientId                 string
	ClientSecret             string
	AuthorizeUrl             string
	TokenUrl                 string
	RedirectUrl              string
	Scope                    string
	ErrorsInStatusCode       bool
	SendClientSecretInParams bool
	UseGetAccessRequest      bool
}
