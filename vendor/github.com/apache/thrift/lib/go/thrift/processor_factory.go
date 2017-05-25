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

// The default processor factory just returns a singleton
// instance.
type TProcessorFactory interface {
	GetProcessor(trans TTransport) TProcessor
}

type tProcessorFactory struct {
	processor TProcessor
}

func NewTProcessorFactory(p TProcessor) TProcessorFactory {
	return &tProcessorFactory{processor: p}
}

func (p *tProcessorFactory) GetProcessor(trans TTransport) TProcessor {
	return p.processor
}

/**
 * The default processor factory just returns a singleton
 * instance.
 */
type TProcessorFunctionFactory interface {
	GetProcessorFunction(trans TTransport) TProcessorFunction
}

type tProcessorFunctionFactory struct {
	processor TProcessorFunction
}

func NewTProcessorFunctionFactory(p TProcessorFunction) TProcessorFunctionFactory {
	return &tProcessorFunctionFactory{processor: p}
}

func (p *tProcessorFunctionFactory) GetProcessorFunction(trans TTransport) TProcessorFunction {
	return p.processor
}
