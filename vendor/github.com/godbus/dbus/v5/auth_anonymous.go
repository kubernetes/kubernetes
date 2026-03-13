package dbus

// AuthAnonymous returns an Auth that uses the ANONYMOUS mechanism.
func AuthAnonymous() Auth {
	return &authAnonymous{}
}

type authAnonymous struct{}

func (a *authAnonymous) FirstData() (name, resp []byte, status AuthStatus) {
	return []byte("ANONYMOUS"), nil, AuthOk
}

func (a *authAnonymous) HandleData(data []byte) (resp []byte, status AuthStatus) {
	return nil, AuthError
}
