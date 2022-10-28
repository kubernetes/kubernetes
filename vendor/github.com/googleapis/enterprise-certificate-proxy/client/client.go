// Copyright 2022 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package client is a cross-platform client for the signer binary (a.k.a."EnterpriseCertSigner").
//
// The signer binary is OS-specific, but exposes a standard set of APIs for the client to use.
package client

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/rsa"
	"crypto/x509"
	"encoding/gob"
	"fmt"
	"io"
	"net/rpc"
	"os"
	"os/exec"

	"github.com/googleapis/enterprise-certificate-proxy/client/util"
)

const signAPI = "EnterpriseCertSigner.Sign"
const certificateChainAPI = "EnterpriseCertSigner.CertificateChain"
const publicKeyAPI = "EnterpriseCertSigner.Public"

// A Connection wraps a pair of unidirectional streams as an io.ReadWriteCloser.
type Connection struct {
	io.ReadCloser
	io.WriteCloser
}

// Close closes c's underlying ReadCloser and WriteCloser.
func (c *Connection) Close() error {
	rerr := c.ReadCloser.Close()
	werr := c.WriteCloser.Close()
	if rerr != nil {
		return rerr
	}
	return werr
}

func init() {
	gob.Register(crypto.SHA256)
	gob.Register(&rsa.PSSOptions{})
}

// SignArgs contains arguments to a crypto Signer.Sign method.
type SignArgs struct {
	Digest []byte            // The content to sign.
	Opts   crypto.SignerOpts // Options for signing, such as Hash identifier.
}

// Key implements credential.Credential by holding the executed signer subprocess.
type Key struct {
	cmd       *exec.Cmd        // Pointer to the signer subprocess.
	client    *rpc.Client      // Pointer to the rpc client that communicates with the signer subprocess.
	publicKey crypto.PublicKey // Public key of loaded certificate.
	chain     [][]byte         // Certificate chain of loaded certificate.
}

// CertificateChain returns the credential as a raw X509 cert chain. This contains the public key.
func (k *Key) CertificateChain() [][]byte {
	return k.chain
}

// Close closes the RPC connection and kills the signer subprocess.
// Call this to free up resources when the Key object is no longer needed.
func (k *Key) Close() error {
	if err := k.cmd.Process.Kill(); err != nil {
		return fmt.Errorf("failed to kill signer process: %w", err)
	}
	if err := k.cmd.Wait(); err.Error() != "signal: killed" {
		return fmt.Errorf("signer process was not killed: %w", err)
	}
	// The Pipes connecting the RPC client should have been closed when the signer subprocess was killed.
	// Calling `k.client.Close()` before `k.cmd.Process.Kill()` or `k.cmd.Wait()` _will_ cause a segfault.
	if err := k.client.Close(); err.Error() != "close |0: file already closed" {
		return fmt.Errorf("failed to close RPC connection: %w", err)
	}
	return nil
}

// Public returns the public key for this Key.
func (k *Key) Public() crypto.PublicKey {
	return k.publicKey
}

// Sign signs a message digest, using the specified signer options.
func (k *Key) Sign(_ io.Reader, digest []byte, opts crypto.SignerOpts) (signed []byte, err error) {
	if opts != nil && opts.HashFunc() != 0 && len(digest) != opts.HashFunc().Size() {
		return nil, fmt.Errorf("Digest length of %v bytes does not match Hash function size of %v bytes", len(digest), opts.HashFunc().Size())
	}
	err = k.client.Call(signAPI, SignArgs{Digest: digest, Opts: opts}, &signed)
	return
}

// Cred spawns a signer subprocess that listens on stdin/stdout to perform certificate
// related operations, including signing messages with the private key.
//
// The signer binary path is read from the specified configFilePath, if provided.
// Otherwise, use the default config file path.
//
// The config file also specifies which certificate the signer should use.
func Cred(configFilePath string) (*Key, error) {
	if configFilePath == "" {
		configFilePath = util.GetDefaultConfigFilePath()
	}
	enterpriseCertSignerPath, err := util.LoadSignerBinaryPath(configFilePath)
	if err != nil {
		return nil, err
	}
	k := &Key{
		cmd: exec.Command(enterpriseCertSignerPath, configFilePath),
	}

	// Redirect errors from subprocess to parent process.
	k.cmd.Stderr = os.Stderr

	// RPC client will communicate with subprocess over stdin/stdout.
	kin, err := k.cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	kout, err := k.cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	k.client = rpc.NewClient(&Connection{kout, kin})

	if err := k.cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting enterprise cert signer subprocess: %w", err)
	}

	if err := k.client.Call(certificateChainAPI, struct{}{}, &k.chain); err != nil {
		return nil, fmt.Errorf("failed to retrieve certificate chain: %w", err)
	}

	var publicKeyBytes []byte
	if err := k.client.Call(publicKeyAPI, struct{}{}, &publicKeyBytes); err != nil {
		return nil, fmt.Errorf("failed to retrieve public key: %w", err)
	}

	publicKey, err := x509.ParsePKIXPublicKey(publicKeyBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to parse public key: %w", err)
	}

	var ok bool
	k.publicKey, ok = publicKey.(crypto.PublicKey)
	if !ok {
		return nil, fmt.Errorf("invalid public key type: %T", publicKey)
	}

	switch pub := k.publicKey.(type) {
	case *rsa.PublicKey:
		if pub.Size() < 256 {
			return nil, fmt.Errorf("RSA modulus size is less than 2048 bits: %v", pub.Size()*8)
		}
	case *ecdsa.PublicKey:
	default:
		return nil, fmt.Errorf("unsupported public key type: %v", pub)
	}

	return k, nil
}
