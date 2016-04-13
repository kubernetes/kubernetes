// Copyright (c) 2014 The SkyDNS Authors. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

// Package server provides a DNS server implementation that handles DNS
// queries. To answer a query, the server asks the provided Backend for
// DNS records, which are then converted to the proper answers.
package server
