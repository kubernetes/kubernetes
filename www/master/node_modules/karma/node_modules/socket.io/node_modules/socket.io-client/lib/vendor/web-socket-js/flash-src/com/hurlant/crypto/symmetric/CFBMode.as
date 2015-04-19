/**
 * CFBMode
 * 
 * An ActionScript 3 implementation of the CFB confidentiality mode
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.symmetric
{
	import flash.utils.ByteArray;

	/**
	 * This is the "full" CFB.
	 * CFB1 and CFB8 are hiding somewhere else.
	 * 
	 * Note: The constructor accepts an optional padding argument, but ignores it otherwise.
	 */
	public class CFBMode extends IVMode implements IMode
	{
		
		public function CFBMode(key:ISymmetricKey, padding:IPad = null) {
			super(key,null);
		}

		public function encrypt(src:ByteArray):void
		{
			var l:uint = src.length;
			var vector:ByteArray = getIV4e();
			for (var i:uint=0;i<src.length;i+=blockSize) {
				key.encrypt(vector);
				var chunk:uint = (i+blockSize<l)?blockSize:l-i;
				for (var j:uint=0;j<chunk;j++) {
					src[i+j] ^= vector[j];
				}
				vector.position=0;
				vector.writeBytes(src, i, chunk);
			}
		}
		
		public function decrypt(src:ByteArray):void
		{
			var l:uint = src.length;
			var vector:ByteArray = getIV4d();
			var tmp:ByteArray = new ByteArray;
			for (var i:uint=0;i<src.length;i+=blockSize) {
				key.encrypt(vector);
				var chunk:uint = (i+blockSize<l)?blockSize:l-i;
				tmp.position=0;
				tmp.writeBytes(src, i, chunk);
				for (var j:uint=0;j<chunk;j++) {
					src[i+j] ^= vector[j];
				}
				vector.position=0;
				vector.writeBytes(tmp);
			}
		}
		
		public function toString():String {
			return key.toString()+"-cfb";
		}

	}
}