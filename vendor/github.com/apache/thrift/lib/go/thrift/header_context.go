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
	"context"
)

// See https://godoc.org/context#WithValue on why do we need the unexported typedefs.
type (
	headerKey     string
	headerKeyList int
)

// Values for headerKeyList.
const (
	headerKeyListRead headerKeyList = iota
	headerKeyListWrite
)

// SetHeader sets a header in the context.
func SetHeader(ctx context.Context, key, value string) context.Context {
	return context.WithValue(
		ctx,
		headerKey(key),
		value,
	)
}

// GetHeader returns a value of the given header from the context.
func GetHeader(ctx context.Context, key string) (value string, ok bool) {
	if v := ctx.Value(headerKey(key)); v != nil {
		value, ok = v.(string)
	}
	return
}

// SetReadHeaderList sets the key list of read THeaders in the context.
func SetReadHeaderList(ctx context.Context, keys []string) context.Context {
	return context.WithValue(
		ctx,
		headerKeyListRead,
		keys,
	)
}

// GetReadHeaderList returns the key list of read THeaders from the context.
func GetReadHeaderList(ctx context.Context) []string {
	if v := ctx.Value(headerKeyListRead); v != nil {
		if value, ok := v.([]string); ok {
			return value
		}
	}
	return nil
}

// SetWriteHeaderList sets the key list of THeaders to write in the context.
func SetWriteHeaderList(ctx context.Context, keys []string) context.Context {
	return context.WithValue(
		ctx,
		headerKeyListWrite,
		keys,
	)
}

// GetWriteHeaderList returns the key list of THeaders to write from the context.
func GetWriteHeaderList(ctx context.Context) []string {
	if v := ctx.Value(headerKeyListWrite); v != nil {
		if value, ok := v.([]string); ok {
			return value
		}
	}
	return nil
}

// AddReadTHeaderToContext adds the whole THeader headers into context.
func AddReadTHeaderToContext(ctx context.Context, headers THeaderMap) context.Context {
	keys := make([]string, 0, len(headers))
	for key, value := range headers {
		ctx = SetHeader(ctx, key, value)
		keys = append(keys, key)
	}
	return SetReadHeaderList(ctx, keys)
}
