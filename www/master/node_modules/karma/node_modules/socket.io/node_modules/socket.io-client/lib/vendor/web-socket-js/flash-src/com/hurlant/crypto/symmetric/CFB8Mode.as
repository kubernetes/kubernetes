/**
 * CFB8Mode
 * 
 * An ActionScript 3 implementation of the CFB-8 confidentiality mode
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.symmetric
{
	import com.hurlant.crypto.tests.TestCase;
	import flash.utils.ByteArray;

	/**
	 * 
	 * Note: The constructor accepts an optional padding argument, but ignores it otherwise.
	 */
	public class CFB8Mode extends IVMode implements IMode
	{
		public function CFB8Mode(key:ISymmetricKey, padding:IPad = null) {
			super(key, null);
		}
		
		public function encrypt(src:ByteArray):void {
			var vector:ByteArray = getIV4e();
			var tmp:ByteArray = new ByteArray;
			for (var i:uint=0;i<src.length;i++) {
				tmp.position = 0;
				tmp.writeBytes(vector);
				key.encrypt(vector);
				src[i] ^= vector[0];
				// rotate
				for (var j:uint=0;j<blockSize-1;j++) {
					vector[j] = tmp[j+1];
				}
				vector[blockSize-1] = src[i];
			}
		}
		
		public function decrypt(src:ByteArray):void {
			var vector:ByteArray = getIV4d();
			var tmp:ByteArray = new ByteArray;
			for (var i:uint=0;i<src.length;i++) {
				var c:uint = src[i];
				tmp.position = 0;
				tmp.writeBytes(vector); // I <- tmp
				key.encrypt(vector);    // O <- vector
				src[i] ^= vector[0];
				// rotate
				for (var j:uint=0;j<blockSize-1;j++) {
					vector[j] = tmp[j+1];
				}
				vector[blockSize-1] = c;
			}

		}
		public function toString():String {
			return key.toString()+"-cfb8";
		}
	}
}