package handshake

import (
	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/qtls"
)

const (
	quicTLSExtensionTypeOldDrafts = 0xffa5
	quicTLSExtensionType          = 0x39
)

type extensionHandler struct {
	ourParams  []byte
	paramsChan chan []byte

	extensionType uint16

	perspective protocol.Perspective
}

var _ tlsExtensionHandler = &extensionHandler{}

// newExtensionHandler creates a new extension handler
func newExtensionHandler(params []byte, pers protocol.Perspective, v protocol.VersionNumber) tlsExtensionHandler {
	et := uint16(quicTLSExtensionType)
	if v != protocol.Version1 {
		et = quicTLSExtensionTypeOldDrafts
	}
	return &extensionHandler{
		ourParams:     params,
		paramsChan:    make(chan []byte),
		perspective:   pers,
		extensionType: et,
	}
}

func (h *extensionHandler) GetExtensions(msgType uint8) []qtls.Extension {
	if (h.perspective == protocol.PerspectiveClient && messageType(msgType) != typeClientHello) ||
		(h.perspective == protocol.PerspectiveServer && messageType(msgType) != typeEncryptedExtensions) {
		return nil
	}
	return []qtls.Extension{{
		Type: h.extensionType,
		Data: h.ourParams,
	}}
}

func (h *extensionHandler) ReceivedExtensions(msgType uint8, exts []qtls.Extension) {
	if (h.perspective == protocol.PerspectiveClient && messageType(msgType) != typeEncryptedExtensions) ||
		(h.perspective == protocol.PerspectiveServer && messageType(msgType) != typeClientHello) {
		return
	}

	var data []byte
	for _, ext := range exts {
		if ext.Type == h.extensionType {
			data = ext.Data
			break
		}
	}

	h.paramsChan <- data
}

func (h *extensionHandler) TransportParameters() <-chan []byte {
	return h.paramsChan
}
