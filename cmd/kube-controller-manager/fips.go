//go:build fips

package main

// enforce fips compliance if boringcrypto is enabled
import _ "crypto/tls/fipsonly"
