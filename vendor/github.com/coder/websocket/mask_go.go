//go:build !amd64 && !arm64 && !js

package websocket

func mask(b []byte, key uint32) uint32 {
	return maskGo(b, key)
}
