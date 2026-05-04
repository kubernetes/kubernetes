// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package json

import (
	"encoding/json"
	"sync"

	"github.com/go-openapi/swag/jsonutils/adapters/ifaces"
)

type adaptersPool struct {
	sync.Pool
}

func (p *adaptersPool) Borrow() *Adapter {
	return p.Get().(*Adapter)
}

func (p *adaptersPool) BorrowIface() ifaces.Adapter {
	return p.Get().(*Adapter)
}

func (p *adaptersPool) Redeem(a *Adapter) {
	p.Put(a)
}

type writersPool struct {
	sync.Pool
}

func (p *writersPool) Borrow() *jwriter {
	ptr := p.Get()

	jw := ptr.(*jwriter)
	jw.Reset()

	return jw
}

func (p *writersPool) Redeem(w *jwriter) {
	p.Put(w)
}

type lexersPool struct {
	sync.Pool
}

func (p *lexersPool) Borrow(data []byte) *jlexer {
	ptr := p.Get()

	l := ptr.(*jlexer)
	l.buf = poolOfReaders.Borrow(data)
	l.dec = json.NewDecoder(l.buf) // cannot pool, not exposed by the encoding/json API
	l.Reset()

	return l
}

func (p *lexersPool) Redeem(l *jlexer) {
	l.dec = nil
	discard := l.buf
	l.buf = nil
	poolOfReaders.Redeem(discard)
	p.Put(l)
}

type readersPool struct {
	sync.Pool
}

func (p *readersPool) Borrow(data []byte) *bytesReader {
	ptr := p.Get()

	b := ptr.(*bytesReader)
	b.Reset()
	b.buf = data

	return b
}

func (p *readersPool) Redeem(b *bytesReader) {
	p.Put(b)
}

var (
	poolOfAdapters = &adaptersPool{
		Pool: sync.Pool{
			New: func() any {
				return NewAdapter()
			},
		},
	}

	poolOfWriters = &writersPool{
		Pool: sync.Pool{
			New: func() any {
				return newJWriter()
			},
		},
	}

	poolOfLexers = &lexersPool{
		Pool: sync.Pool{
			New: func() any {
				return newLexer(nil)
			},
		},
	}

	poolOfReaders = &readersPool{
		Pool: sync.Pool{
			New: func() any {
				return &bytesReader{}
			},
		},
	}
)

// BorrowAdapter borrows an [Adapter] from the pool, recycling already allocated instances.
func BorrowAdapter() *Adapter {
	return poolOfAdapters.Borrow()
}

// BorrowAdapterIface borrows a stdlib [Adapter] and converts it directly
// to [ifaces.Adapter]. This is useful to avoid further allocations when
// translating the concrete type into an interface.
func BorrowAdapterIface() ifaces.Adapter {
	return poolOfAdapters.BorrowIface()
}

// RedeemAdapter redeems an [Adapter] to the pool, so it may be recycled.
func RedeemAdapter(a *Adapter) {
	poolOfAdapters.Redeem(a)
}

func RedeemAdapterIface(a ifaces.Adapter) {
	concrete, ok := a.(*Adapter)
	if ok {
		poolOfAdapters.Redeem(concrete)
	}
}
