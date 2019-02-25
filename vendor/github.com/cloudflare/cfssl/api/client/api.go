package client

// SignResult is the result of signing a CSR.
type SignResult struct {
	Certificate []byte `json:"certificate"`
}
