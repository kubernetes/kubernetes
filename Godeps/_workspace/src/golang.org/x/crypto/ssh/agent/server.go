// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agent

import (
	"crypto/rsa"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math/big"

	"golang.org/x/crypto/ssh"
)

// Server wraps an Agent and uses it to implement the agent side of
// the SSH-agent, wire protocol.
type server struct {
	agent Agent
}

func (s *server) processRequestBytes(reqData []byte) []byte {
	rep, err := s.processRequest(reqData)
	if err != nil {
		if err != errLocked {
			// TODO(hanwen): provide better logging interface?
			log.Printf("agent %d: %v", reqData[0], err)
		}
		return []byte{agentFailure}
	}

	if err == nil && rep == nil {
		return []byte{agentSuccess}
	}

	return ssh.Marshal(rep)
}

func marshalKey(k *Key) []byte {
	var record struct {
		Blob    []byte
		Comment string
	}
	record.Blob = k.Marshal()
	record.Comment = k.Comment

	return ssh.Marshal(&record)
}

type agentV1IdentityMsg struct {
	Numkeys uint32 `sshtype:"2"`
}

type agentRemoveIdentityMsg struct {
	KeyBlob []byte `sshtype:"18"`
}

type agentLockMsg struct {
	Passphrase []byte `sshtype:"22"`
}

type agentUnlockMsg struct {
	Passphrase []byte `sshtype:"23"`
}

func (s *server) processRequest(data []byte) (interface{}, error) {
	switch data[0] {
	case agentRequestV1Identities:
		return &agentV1IdentityMsg{0}, nil
	case agentRemoveIdentity:
		var req agentRemoveIdentityMsg
		if err := ssh.Unmarshal(data, &req); err != nil {
			return nil, err
		}

		var wk wireKey
		if err := ssh.Unmarshal(req.KeyBlob, &wk); err != nil {
			return nil, err
		}

		return nil, s.agent.Remove(&Key{Format: wk.Format, Blob: req.KeyBlob})

	case agentRemoveAllIdentities:
		return nil, s.agent.RemoveAll()

	case agentLock:
		var req agentLockMsg
		if err := ssh.Unmarshal(data, &req); err != nil {
			return nil, err
		}

		return nil, s.agent.Lock(req.Passphrase)

	case agentUnlock:
		var req agentLockMsg
		if err := ssh.Unmarshal(data, &req); err != nil {
			return nil, err
		}
		return nil, s.agent.Unlock(req.Passphrase)

	case agentSignRequest:
		var req signRequestAgentMsg
		if err := ssh.Unmarshal(data, &req); err != nil {
			return nil, err
		}

		var wk wireKey
		if err := ssh.Unmarshal(req.KeyBlob, &wk); err != nil {
			return nil, err
		}

		k := &Key{
			Format: wk.Format,
			Blob:   req.KeyBlob,
		}

		sig, err := s.agent.Sign(k, req.Data) //  TODO(hanwen): flags.
		if err != nil {
			return nil, err
		}
		return &signResponseAgentMsg{SigBlob: ssh.Marshal(sig)}, nil
	case agentRequestIdentities:
		keys, err := s.agent.List()
		if err != nil {
			return nil, err
		}

		rep := identitiesAnswerAgentMsg{
			NumKeys: uint32(len(keys)),
		}
		for _, k := range keys {
			rep.Keys = append(rep.Keys, marshalKey(k)...)
		}
		return rep, nil
	case agentAddIdentity:
		return nil, s.insertIdentity(data)
	}

	return nil, fmt.Errorf("unknown opcode %d", data[0])
}

func (s *server) insertIdentity(req []byte) error {
	var record struct {
		Type string `sshtype:"17"`
		Rest []byte `ssh:"rest"`
	}
	if err := ssh.Unmarshal(req, &record); err != nil {
		return err
	}

	switch record.Type {
	case ssh.KeyAlgoRSA:
		var k rsaKeyMsg
		if err := ssh.Unmarshal(req, &k); err != nil {
			return err
		}

		priv := rsa.PrivateKey{
			PublicKey: rsa.PublicKey{
				E: int(k.E.Int64()),
				N: k.N,
			},
			D:      k.D,
			Primes: []*big.Int{k.P, k.Q},
		}
		priv.Precompute()

		return s.agent.Add(&priv, nil, k.Comments)
	}
	return fmt.Errorf("not implemented: %s", record.Type)
}

// ServeAgent serves the agent protocol on the given connection. It
// returns when an I/O error occurs.
func ServeAgent(agent Agent, c io.ReadWriter) error {
	s := &server{agent}

	var length [4]byte
	for {
		if _, err := io.ReadFull(c, length[:]); err != nil {
			return err
		}
		l := binary.BigEndian.Uint32(length[:])
		if l > maxAgentResponseBytes {
			// We also cap requests.
			return fmt.Errorf("agent: request too large: %d", l)
		}

		req := make([]byte, l)
		if _, err := io.ReadFull(c, req); err != nil {
			return err
		}

		repData := s.processRequestBytes(req)
		if len(repData) > maxAgentResponseBytes {
			return fmt.Errorf("agent: reply too large: %d bytes", len(repData))
		}

		binary.BigEndian.PutUint32(length[:], uint32(len(repData)))
		if _, err := c.Write(length[:]); err != nil {
			return err
		}
		if _, err := c.Write(repData); err != nil {
			return err
		}
	}
}
