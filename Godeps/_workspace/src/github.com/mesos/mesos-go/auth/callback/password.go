package callback

type Password struct {
	password []byte
}

func NewPassword() *Password {
	return &Password{}
}

func (cb *Password) Get() []byte {
	clone := make([]byte, len(cb.password))
	copy(clone, cb.password)
	return clone
}

func (cb *Password) Set(password []byte) {
	cb.password = make([]byte, len(password))
	copy(cb.password, password)
}
