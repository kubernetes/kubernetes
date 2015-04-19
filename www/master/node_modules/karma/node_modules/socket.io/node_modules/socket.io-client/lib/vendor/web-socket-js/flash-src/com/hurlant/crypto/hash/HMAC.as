/**
 * HMAC
 * 
 * An ActionScript 3 implementation of HMAC, Keyed-Hashing for Message
 * Authentication, as defined by RFC-2104
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.hash
{
	import flash.utils.ByteArray;
	import com.hurlant.util.Hex;
	
	public class HMAC implements IHMAC
	{
		private var hash:IHash;
		private var bits:uint;
		
		/**
		 * Create a HMAC object, using a Hash function, and 
		 * optionally a number of bits to return. 
		 * The HMAC will be truncated to that size if needed.
		 */
		public function HMAC(hash:IHash, bits:uint=0) {
			this.hash = hash;
			this.bits = bits;
		}
		

		public function getHashSize():uint {
			if (bits!=0) {
				return bits/8;
			} else {
				return hash.getHashSize();
			}
		}
		
		/**
		 * Compute a HMAC using a key and some data.
		 * It doesn't modify either, and returns a new ByteArray with the HMAC value.
		 */
		public function compute(key:ByteArray, data:ByteArray):ByteArray {
			var hashKey:ByteArray;
			if (key.length>hash.getInputSize()) {
				hashKey = hash.hash(key);
			} else {
				hashKey = new ByteArray;
				hashKey.writeBytes(key);
			}
			while (hashKey.length<hash.getInputSize()) {
				hashKey[hashKey.length]=0;
			}
			var innerKey:ByteArray = new ByteArray;
			var outerKey:ByteArray = new ByteArray;
			for (var i:uint=0;i<hashKey.length;i++) {
				innerKey[i] = hashKey[i] ^ 0x36;
				outerKey[i] = hashKey[i] ^ 0x5c;
			}
			// inner + data
			innerKey.position = hashKey.length;
			innerKey.writeBytes(data);
			var innerHash:ByteArray = hash.hash(innerKey);
			// outer + innerHash
			outerKey.position = hashKey.length;
			outerKey.writeBytes(innerHash);
			var outerHash:ByteArray = hash.hash(outerKey);
			if (bits>0 && bits<8*outerHash.length) {
				outerHash.length = bits/8;
			}
			return outerHash;
		}
		public function dispose():void {
			hash = null;
			bits = 0;
		}
		public function toString():String {
			return "hmac-"+(bits>0?bits+"-":"")+hash.toString();
		}
		
	}
}