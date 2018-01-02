package s3crypto

// Padder handles padding of crypto data
type Padder interface {
	// Pad will pad the byte array.
	// The second parameter is NOT how many
	// bytes to pad by, but how many bytes
	// have been read prior to the padding.
	// This allows for streamable padding.
	Pad([]byte, int) ([]byte, error)
	// Unpad will unpad the byte bytes. Unpad
	// methods must be constant time.
	Unpad([]byte) ([]byte, error)
	// Name returns the name of the padder.
	// This is used when decrypting on
	// instantiating new padders.
	Name() string
}

// NoPadder does not pad anything
var NoPadder = Padder(noPadder{})

type noPadder struct{}

func (padder noPadder) Pad(b []byte, n int) ([]byte, error) {
	return b, nil
}

func (padder noPadder) Unpad(b []byte) ([]byte, error) {
	return b, nil
}

func (padder noPadder) Name() string {
	return "NoPadding"
}
