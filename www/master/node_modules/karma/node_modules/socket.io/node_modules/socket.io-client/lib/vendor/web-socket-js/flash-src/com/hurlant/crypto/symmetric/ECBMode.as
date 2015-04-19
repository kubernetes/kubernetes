/**
 * ECBMode
 * 
 * An ActionScript 3 implementation of the ECB confidentiality mode
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.symmetric
{
	import flash.utils.ByteArray;
	import com.hurlant.util.Memory;
	import com.hurlant.util.Hex;
	
	/**
	 * ECB mode.
	 * This uses a padding and a symmetric key.
	 * If no padding is given, PKCS#5 is used.
	 */
	public class ECBMode implements IMode, ICipher
	{
		private var key:ISymmetricKey;
		private var padding:IPad;
		
		public function ECBMode(key:ISymmetricKey, padding:IPad = null) {
			this.key = key;
			if (padding == null) {
				padding = new PKCS5(key.getBlockSize());
			} else {
				padding.setBlockSize(key.getBlockSize());
			}
			this.padding = padding;
		}
		
		public function getBlockSize():uint {
			return key.getBlockSize();
		}
		
		public function encrypt(src:ByteArray):void {
			padding.pad(src);
			src.position = 0;
			var blockSize:uint = key.getBlockSize();
			var tmp:ByteArray = new ByteArray;
			var dst:ByteArray = new ByteArray;
			for (var i:uint=0;i<src.length;i+=blockSize) {
				tmp.length=0;
				src.readBytes(tmp, 0, blockSize);
				key.encrypt(tmp);
				dst.writeBytes(tmp);
			}
			src.length=0;
			src.writeBytes(dst);
		}
		public function decrypt(src:ByteArray):void {
			src.position = 0;
			var blockSize:uint = key.getBlockSize();
			
			// sanity check.
			if (src.length%blockSize!=0) {
				throw new Error("ECB mode cipher length must be a multiple of blocksize "+blockSize);
			}
			
			var tmp:ByteArray = new ByteArray;
			var dst:ByteArray = new ByteArray;
			for (var i:uint=0;i<src.length;i+=blockSize) {
				tmp.length=0;
				src.readBytes(tmp, 0, blockSize);
				
				key.decrypt(tmp);
				dst.writeBytes(tmp);
			}
			padding.unpad(dst);
			src.length=0;
			src.writeBytes(dst);
		}
		public function dispose():void {
			key.dispose();
			key = null;
			padding = null;
			Memory.gc();
		}
		public function toString():String {
			return key.toString()+"-ecb";
		}
	}
}