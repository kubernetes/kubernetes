/**
 * SHA224
 * 
 * An ActionScript 3 implementation of Secure Hash Algorithm, SHA-224, as defined
 * in FIPS PUB 180-2
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.hash
{
	public class SHA224 extends SHA256
	{
		function SHA224() {
			h = [
				0xc1059ed8, 0x367cd507, 0x3070dd17, 0xf70e5939,
				0xffc00b31, 0x68581511, 0x64f98fa7, 0xbefa4fa4
			];
		}
		
		public override function getHashSize():uint {
			return 28;
		}
		public override function toString():String {
			return "sha224";
		}
	}
}