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
	"errors"
)

// Generic Thrift exception
type TException interface {
	error
}

// Prepends additional information to an error without losing the Thrift exception interface
func PrependError(prepend string, err error) error {
	if t, ok := err.(TTransportException); ok {
		return NewTTransportException(t.TypeId(), prepend+t.Error())
	}
	if t, ok := err.(TProtocolException); ok {
		return NewTProtocolExceptionWithType(t.TypeId(), errors.New(prepend+err.Error()))
	}
	if t, ok := err.(TApplicationException); ok {
		return NewTApplicationException(t.TypeId(), prepend+t.Error())
	}

	return errors.New(prepend + err.Error())
}
