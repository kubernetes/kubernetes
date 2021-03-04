// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkcs12

import (
	"bytes"
	"crypto/sha1"
	"math/big"
)

var (
	one = big.NewInt(1)
)

// sha1Sum returns the SHA-1 hash of in.
func sha1Sum(in []byte) []byte {
	sum := sha1.Sum(in)
	return sum[:]
}

// fillWithRepeats returns v*ceiling(len(pattern) / v) bytes consisting of
// repeats of pattern.
func fillWithRepeats(pattern []byte, v int) []byte {
	if len(pattern) == 0 {
		return nil
	}
	outputLen := v * ((len(pattern) + v - 1) / v)
	return bytes.Repeat(pattern, (outputLen+len(pattern)-1)/len(pattern))[:outputLen]
}

func pbkdf(hash func([]byte) []byte, u, v int, salt, password []byte, r int, ID byte, size int) (key []byte) {
	// implementation of https://tools.ietf.org/html/rfc7292#appendix-B.2 , RFC text verbatim in comments

	//    Let H be a hash function built around a compression function f:

	//       Z_2^u x Z_2^v -> Z_2^u

	//    (that is, H has a chaining variable and output of length u bits, and
	//    the message input to the compression function of H is v bits).  The
	//    values for u and v are as follows:

	//            HASH FUNCTION     VALUE u        VALUE v
	//              MD2, MD5          128            512
	//                SHA-1           160            512
	//               SHA-224          224            512
	//               SHA-256          256            512
	//               SHA-384          384            1024
	//               SHA-512          512            1024
	//             SHA-512/224        224            1024
	//             SHA-512/256        256            1024

	//    Furthermore, let r be the iteration count.

	//    We assume here that u and v are both multiples of 8, as are the
	//    lengths of the password and salt strings (which we denote by p and s,
	//    respectively) and the number n of pseudorandom bits required.  In
	//    addition, u and v are of course non-zero.

	//    For information on security considerations for MD5 [19], see [25] and
	//    [1], and on those for MD2, see [18].

	//    The following procedure can be used to produce pseudorandom bits for
	//    a particular "purpose" that is identified by a byte called "ID".
	//    This standard specifies 3 different values for the ID byte:

	//    1.  If ID=1, then the pseudorandom bits being produced are to be used
	//        as key material for performing encryption or decryption.

	//    2.  If ID=2, then the pseudorandom bits being produced are to be used
	//        as an IV (Initial Value) for encryption or decryption.

	//    3.  If ID=3, then the pseudorandom bits being produced are to be used
	//        as an integrity key for MACing.

	//    1.  Construct a string, D (the "diversifier"), by concatenating v/8
	//        copies of ID.
	var D []byte
	for i := 0; i < v; i++ {
		D = append(D, ID)
	}

	//    2.  Concatenate copies of the salt together to create a string S of
	//        length v(ceiling(s/v)) bits (the final copy of the salt may be
	//        truncated to create S).  Note that if the salt is the empty
	//        string, then so is S.

	S := fillWithRepeats(salt, v)

	//    3.  Concatenate copies of the password together to create a string P
	//        of length v(ceiling(p/v)) bits (the final copy of the password
	//        may be truncated to create P).  Note that if the password is the
	//        empty string, then so is P.

	P := fillWithRepeats(password, v)

	//    4.  Set I=S||P to be the concatenation of S and P.
	I := append(S, P...)

	//    5.  Set c=ceiling(n/u).
	c := (size + u - 1) / u

	//    6.  For i=1, 2, ..., c, do the following:
	A := make([]byte, c*20)
	var IjBuf []byte
	for i := 0; i < c; i++ {
		//        A.  Set A2=H^r(D||I). (i.e., the r-th hash of D||1,
		//            H(H(H(... H(D||I))))
		Ai := hash(append(D, I...))
		for j := 1; j < r; j++ {
			Ai = hash(Ai)
		}
		copy(A[i*20:], Ai[:])

		if i < c-1 { // skip on last iteration
			// B.  Concatenate copies of Ai to create a string B of length v
			//     bits (the final copy of Ai may be truncated to create B).
			var B []byte
			for len(B) < v {
				B = append(B, Ai[:]...)
			}
			B = B[:v]

			// C.  Treating I as a concatenation I_0, I_1, ..., I_(k-1) of v-bit
			//     blocks, where k=ceiling(s/v)+ceiling(p/v), modify I by
			//     setting I_j=(I_j+B+1) mod 2^v for each j.
			{
				Bbi := new(big.Int).SetBytes(B)
				Ij := new(big.Int)

				for j := 0; j < len(I)/v; j++ {
					Ij.SetBytes(I[j*v : (j+1)*v])
					Ij.Add(Ij, Bbi)
					Ij.Add(Ij, one)
					Ijb := Ij.Bytes()
					// We expect Ijb to be exactly v bytes,
					// if it is longer or shorter we must
					// adjust it accordingly.
					if len(Ijb) > v {
						Ijb = Ijb[len(Ijb)-v:]
					}
					if len(Ijb) < v {
						if IjBuf == nil {
							IjBuf = make([]byte, v)
						}
						bytesShort := v - len(Ijb)
						for i := 0; i < bytesShort; i++ {
							IjBuf[i] = 0
						}
						copy(IjBuf[bytesShort:], Ijb)
						Ijb = IjBuf
					}
					copy(I[j*v:(j+1)*v], Ijb)
				}
			}
		}
	}
	//    7.  Concatenate A_1, A_2, ..., A_c together to form a pseudorandom
	//        bit string, A.

	//    8.  Use the first n bits of A as the output of this entire process.
	return A[:size]

	//    If the above process is being used to generate a DES key, the process
	//    should be used to create 64 random bits, and the key's parity bits
	//    should be set after the 64 bits have been produced.  Similar concerns
	//    hold for 2-key and 3-key triple-DES keys, for CDMF keys, and for any
	//    similar keys with parity bits "built into them".
}
