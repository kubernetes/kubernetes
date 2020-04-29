/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package thrift

import (
	"log"
	"runtime/debug"
	"sync"
	"sync/atomic"
)

/*
 * This is not a typical TSimpleServer as it is not blocked after accept a socket.
 * It is more like a TThreadedServer that can handle different connections in different goroutines.
 * This will work if golang user implements a conn-pool like thing in client side.
 */
type TSimpleServer struct {
	closed int32
	wg     sync.WaitGroup
	mu     sync.Mutex

	processorFactory       TProcessorFactory
	serverTransport        TServerTransport
	inputTransportFactory  TTransportFactory
	outputTransportFactory TTransportFactory
	inputProtocolFactory   TProtocolFactory
	outputProtocolFactory  TProtocolFactory

	// Headers to auto forward in THeaderProtocol
	forwardHeaders []string
}

func NewTSimpleServer2(processor TProcessor, serverTransport TServerTransport) *TSimpleServer {
	return NewTSimpleServerFactory2(NewTProcessorFactory(processor), serverTransport)
}

func NewTSimpleServer4(processor TProcessor, serverTransport TServerTransport, transportFactory TTransportFactory, protocolFactory TProtocolFactory) *TSimpleServer {
	return NewTSimpleServerFactory4(NewTProcessorFactory(processor),
		serverTransport,
		transportFactory,
		protocolFactory,
	)
}

func NewTSimpleServer6(processor TProcessor, serverTransport TServerTransport, inputTransportFactory TTransportFactory, outputTransportFactory TTransportFactory, inputProtocolFactory TProtocolFactory, outputProtocolFactory TProtocolFactory) *TSimpleServer {
	return NewTSimpleServerFactory6(NewTProcessorFactory(processor),
		serverTransport,
		inputTransportFactory,
		outputTransportFactory,
		inputProtocolFactory,
		outputProtocolFactory,
	)
}

func NewTSimpleServerFactory2(processorFactory TProcessorFactory, serverTransport TServerTransport) *TSimpleServer {
	return NewTSimpleServerFactory6(processorFactory,
		serverTransport,
		NewTTransportFactory(),
		NewTTransportFactory(),
		NewTBinaryProtocolFactoryDefault(),
		NewTBinaryProtocolFactoryDefault(),
	)
}

func NewTSimpleServerFactory4(processorFactory TProcessorFactory, serverTransport TServerTransport, transportFactory TTransportFactory, protocolFactory TProtocolFactory) *TSimpleServer {
	return NewTSimpleServerFactory6(processorFactory,
		serverTransport,
		transportFactory,
		transportFactory,
		protocolFactory,
		protocolFactory,
	)
}

func NewTSimpleServerFactory6(processorFactory TProcessorFactory, serverTransport TServerTransport, inputTransportFactory TTransportFactory, outputTransportFactory TTransportFactory, inputProtocolFactory TProtocolFactory, outputProtocolFactory TProtocolFactory) *TSimpleServer {
	return &TSimpleServer{
		processorFactory:       processorFactory,
		serverTransport:        serverTransport,
		inputTransportFactory:  inputTransportFactory,
		outputTransportFactory: outputTransportFactory,
		inputProtocolFactory:   inputProtocolFactory,
		outputProtocolFactory:  outputProtocolFactory,
	}
}

func (p *TSimpleServer) ProcessorFactory() TProcessorFactory {
	return p.processorFactory
}

func (p *TSimpleServer) ServerTransport() TServerTransport {
	return p.serverTransport
}

func (p *TSimpleServer) InputTransportFactory() TTransportFactory {
	return p.inputTransportFactory
}

func (p *TSimpleServer) OutputTransportFactory() TTransportFactory {
	return p.outputTransportFactory
}

func (p *TSimpleServer) InputProtocolFactory() TProtocolFactory {
	return p.inputProtocolFactory
}

func (p *TSimpleServer) OutputProtocolFactory() TProtocolFactory {
	return p.outputProtocolFactory
}

func (p *TSimpleServer) Listen() error {
	return p.serverTransport.Listen()
}

// SetForwardHeaders sets the list of header keys that will be auto forwarded
// while using THeaderProtocol.
//
// "forward" means that when the server is also a client to other upstream
// thrift servers, the context object user gets in the processor functions will
// have both read and write headers set, with write headers being forwarded.
// Users can always override the write headers by calling SetWriteHeaderList
// before calling thrift client functions.
func (p *TSimpleServer) SetForwardHeaders(headers []string) {
	size := len(headers)
	if size == 0 {
		p.forwardHeaders = nil
		return
	}

	keys := make([]string, size)
	copy(keys, headers)
	p.forwardHeaders = keys
}

func (p *TSimpleServer) innerAccept() (int32, error) {
	client, err := p.serverTransport.Accept()
	p.mu.Lock()
	defer p.mu.Unlock()
	closed := atomic.LoadInt32(&p.closed)
	if closed != 0 {
		return closed, nil
	}
	if err != nil {
		return 0, err
	}
	if client != nil {
		p.wg.Add(1)
		go func() {
			defer p.wg.Done()
			if err := p.processRequests(client); err != nil {
				log.Println("error processing request:", err)
			}
		}()
	}
	return 0, nil
}

func (p *TSimpleServer) AcceptLoop() error {
	for {
		closed, err := p.innerAccept()
		if err != nil {
			return err
		}
		if closed != 0 {
			return nil
		}
	}
}

func (p *TSimpleServer) Serve() error {
	err := p.Listen()
	if err != nil {
		return err
	}
	p.AcceptLoop()
	return nil
}

func (p *TSimpleServer) Stop() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if atomic.LoadInt32(&p.closed) != 0 {
		return nil
	}
	atomic.StoreInt32(&p.closed, 1)
	p.serverTransport.Interrupt()
	p.wg.Wait()
	return nil
}

func (p *TSimpleServer) processRequests(client TTransport) error {
	processor := p.processorFactory.GetProcessor(client)
	inputTransport, err := p.inputTransportFactory.GetTransport(client)
	if err != nil {
		return err
	}
	inputProtocol := p.inputProtocolFactory.GetProtocol(inputTransport)
	var outputTransport TTransport
	var outputProtocol TProtocol

	// for THeaderProtocol, we must use the same protocol instance for
	// input and output so that the response is in the same dialect that
	// the server detected the request was in.
	headerProtocol, ok := inputProtocol.(*THeaderProtocol)
	if ok {
		outputProtocol = inputProtocol
	} else {
		oTrans, err := p.outputTransportFactory.GetTransport(client)
		if err != nil {
			return err
		}
		outputTransport = oTrans
		outputProtocol = p.outputProtocolFactory.GetProtocol(outputTransport)
	}

	defer func() {
		if e := recover(); e != nil {
			log.Printf("panic in processor: %s: %s", e, debug.Stack())
		}
	}()

	if inputTransport != nil {
		defer inputTransport.Close()
	}
	if outputTransport != nil {
		defer outputTransport.Close()
	}
	for {
		if atomic.LoadInt32(&p.closed) != 0 {
			return nil
		}

		ctx := defaultCtx
		if headerProtocol != nil {
			// We need to call ReadFrame here, otherwise we won't
			// get any headers on the AddReadTHeaderToContext call.
			//
			// ReadFrame is safe to be called multiple times so it
			// won't break when it's called again later when we
			// actually start to read the message.
			if err := headerProtocol.ReadFrame(); err != nil {
				return err
			}
			ctx = AddReadTHeaderToContext(defaultCtx, headerProtocol.GetReadHeaders())
			ctx = SetWriteHeaderList(ctx, p.forwardHeaders)
		}

		ok, err := processor.Process(ctx, inputProtocol, outputProtocol)
		if err, ok := err.(TTransportException); ok && err.TypeId() == END_OF_FILE {
			return nil
		} else if err != nil {
			return err
		}
		if err, ok := err.(TApplicationException); ok && err.TypeId() == UNKNOWN_METHOD {
			continue
		}
		if !ok {
			break
		}
	}
	return nil
}
