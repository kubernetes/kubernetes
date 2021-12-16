package protocol

// Perspective determines if we're acting as a server or a client
type Perspective int

// the perspectives
const (
	PerspectiveServer Perspective = 1
	PerspectiveClient Perspective = 2
)

// Opposite returns the perspective of the peer
func (p Perspective) Opposite() Perspective {
	return 3 - p
}

func (p Perspective) String() string {
	switch p {
	case PerspectiveServer:
		return "Server"
	case PerspectiveClient:
		return "Client"
	default:
		return "invalid perspective"
	}
}
