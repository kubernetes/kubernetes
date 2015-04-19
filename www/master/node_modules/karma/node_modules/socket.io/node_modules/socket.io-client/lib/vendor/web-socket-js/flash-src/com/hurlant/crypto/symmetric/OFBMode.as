/**
 * OFBMode
 * 
 * An ActionScript 3 implementation of the OFB confidentiality mode
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.symmetric
{
	import flash.utils.ByteArray;

	public class OFBMode extends IVMode implements IMode
	{
		public function OFBMode(key:ISymmetricKey, padding:IPad=null)
		{
			super(key, null);
		}
		
		public function encrypt(src:ByteArray):void
		{
			var vector:ByteArray = getIV4e();
			core(src, vector);
		}
		
		public function decrypt(src:ByteArray):void
		{
			var vector:ByteArray = getIV4d();
			core(src, vector);
		}
		
		private function core(src:ByteArray, iv:ByteArray):void { 
			var l:uint = src.length;
			var tmp:ByteArray = new ByteArray;
			for (var i:uint=0;i<src.length;i+=blockSize) {
				key.encrypt(iv);
				tmp.position=0;
				tmp.writeBytes(iv);
				var chunk:uint = (i+blockSize<l)?blockSize:l-i;
				for (var j:uint=0;j<chunk;j++) {
					src[i+j] ^= iv[j];
				}
				iv.position=0;
				iv.writeBytes(tmp);
			}
		}
		public function toString():String {
			return key.toString()+"-ofb";
		}
		
	}
}