package tls

// SayHello constructs a simple Client Hello to a server, parses its serverHelloMsg response
// and returns the negotiated ciphersuite ID, and, if an EC cipher suite, the curve ID
func (c *Conn) SayHello(newSigAls []SignatureAndHash) (cipherID, curveType uint16, curveID CurveID, version uint16, certs [][]byte, err error) {
	// Set the supported signatures and hashes to the set `newSigAls`
	supportedSignatureAlgorithms := make([]signatureAndHash, len(newSigAls))
	for i := range newSigAls {
		supportedSignatureAlgorithms[i] = newSigAls[i].internal()
	}

	hello := &clientHelloMsg{
		vers:                c.config.maxVersion(),
		compressionMethods:  []uint8{compressionNone},
		random:              make([]byte, 32),
		ocspStapling:        true,
		serverName:          c.config.ServerName,
		supportedCurves:     c.config.curvePreferences(),
		supportedPoints:     []uint8{pointFormatUncompressed},
		nextProtoNeg:        len(c.config.NextProtos) > 0,
		secureRenegotiation: true,
		cipherSuites:        c.config.cipherSuites(),
		signatureAndHashes:  supportedSignatureAlgorithms,
	}
	serverHello, err := c.sayHello(hello)
	if err != nil {
		return
	}
	// Prime the connection, if necessary, for key
	// exchange messages by reading off the certificate
	// message and, if necessary, the OCSP stapling
	// message
	var msg interface{}
	msg, err = c.readHandshake()
	if err != nil {
		return
	}
	certMsg, ok := msg.(*certificateMsg)
	if !ok || len(certMsg.certificates) == 0 {
		err = unexpectedMessageError(certMsg, msg)
		return
	}
	certs = certMsg.certificates

	if serverHello.ocspStapling {
		msg, err = c.readHandshake()
		if err != nil {
			return
		}
		certStatusMsg, ok := msg.(*certificateStatusMsg)
		if !ok {
			err = unexpectedMessageError(certStatusMsg, msg)
			return
		}
	}

	if CipherSuites[serverHello.cipherSuite].EllipticCurve {

		var skx *serverKeyExchangeMsg
		skx, err = c.exchangeKeys()
		if err != nil {
			return
		}
		if skx.raw[0] != typeServerKeyExchange {
			err = unexpectedMessageError(skx, msg)
			return
		}
		if len(skx.key) < 4 {
			err = unexpectedMessageError(skx, msg)
			return
		}
		curveType = uint16(skx.key[0])
		// If we have a named curve, report which one it is.
		if curveType == 3 {
			curveID = CurveID(skx.key[1])<<8 | CurveID(skx.key[2])
		}
	}
	cipherID, version = serverHello.cipherSuite, serverHello.vers

	return
}

// sayHello is the backend to SayHello that returns a full serverHelloMsg for processing.
func (c *Conn) sayHello(hello *clientHelloMsg) (serverHello *serverHelloMsg, err error) {
	c.writeRecord(recordTypeHandshake, hello.marshal())
	msg, err := c.readHandshake()
	if err != nil {
		return
	}
	serverHello, ok := msg.(*serverHelloMsg)
	if !ok {
		return nil, unexpectedMessageError(serverHello, msg)
	}
	return
}

// exchangeKeys continues the handshake to receive the serverKeyExchange message,
// from which we can extract elliptic curve parameters
func (c *Conn) exchangeKeys() (serverKeyExchange *serverKeyExchangeMsg, err error) {
	msg, err := c.readHandshake()
	if err != nil {
		return
	}
	serverKeyExchange, ok := msg.(*serverKeyExchangeMsg)
	if !ok {
		return nil, unexpectedMessageError(serverKeyExchange, msg)
	}
	return
}
