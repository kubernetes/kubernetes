/**
 * SHA256
 * 
 * An ActionScript 3 implementation of Secure Hash Algorithm, SHA-256, as defined
 * in FIPS PUB 180-2
 * Copyright (c) 2007 Henri Torgemane
 * 
 * Derived from:
 * 		A JavaScript implementation of the Secure Hash Standard
 * 		Version 0.3 Copyright Angel Marin 2003-2004 - http://anmar.eu.org/
 * Derived from:
 * 		A JavaScript implementation of the Secure Hash Algorithm, SHA-1, as defined
 * 		in FIPS PUB 180-1
 * 		Version 2.1a Copyright Paul Johnston 2000 - 2002.
 * 		Other contributors: Greg Holt, Andrew Kepert, Ydnar, Lostinet
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.hash
{
	

	public class SHA256 extends SHABase implements IHash
	{
		
		protected static const k:Array = [
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2];
		protected var h:Array = [
		0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
		];
		
		public function SHA256(){
		}
		
		public override function getHashSize():uint
		{
			return 32;
		}
		
		protected override function core(x:Array, len:uint):Array {
			/* append padding */
			x[len >> 5] |= 0x80 << (24 - len % 32);
			x[((len + 64 >> 9) << 4) + 15] = len;
		
			var w:Array = [];
			var a:uint = h[0];
			var b:uint = h[1];
			var c:uint = h[2];
			var d:uint = h[3];
			var e:uint = h[4];
			var f:uint = h[5];
			var g:uint = h[6];
			var h:uint = h[7];
			
			for (var i:uint=0; i<x.length; i+=16) {
				var olda:uint = a;
				var oldb:uint = b;
				var oldc:uint = c;
				var oldd:uint = d;
				var olde:uint = e;
				var oldf:uint = f;
				var oldg:uint = g;
				var oldh:uint = h;

				for (var j:uint=0; j<64; j++) {
					if (j<16) {
						w[j] = x[i+j] || 0;
					} else {
						var s0:uint = rrol(w[j-15],7)^rrol(w[j-15],18)^(w[j-15]>>>3);
						var s1:uint = rrol(w[j-2], 17)^rrol(w[j-2],19)^(w[j-2]>>>10);
						w[j] = w[j-16] + s0 + w[j-7] + s1;
					}
					var t2:uint = (rrol(a,2) ^ rrol(a,13) ^ rrol(a,22)) + ((a&b) ^ (a&c) ^ (b&c));
					var t1:uint = h + (rrol(e,6) ^ rrol(e,11) ^ rrol(e,25)) + ((e&f)^(g&~e)) + k[j] + w[j]
					h = g;
					g = f;
					f = e;
					e = d + t1;
					d = c;
					c = b;
					b = a;
					a = t1 + t2;

				}
				a += olda;
				b += oldb;
				c += oldc;
				d += oldd;
				e += olde;
				f += oldf;
				g += oldg;
				h += oldh;
			}
			return [ a,b,c,d,e,f,g,h ];
		}
		
		/*
		 * Bitwise rotate a 32-bit number to the right.
		 */
		protected function rrol(num:uint, cnt:uint):uint {
		  return (num << (32-cnt)) | (num >>> cnt);
		}
		
		public override function toString():String {
			return "sha256";
		}
	}
}