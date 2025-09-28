package wmi

// Credentials
type Credentials struct {
	UserName string
	Password string
	Domain   string
}

// GetSecureString
func (cred Credentials) GetSecureString() (string, error) {
	panic("not implemented")
}
