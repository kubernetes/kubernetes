/**
 * SimpleIVMode
 * 
 * A convenience class that automatically places the IV
 * at the beginning of the encrypted stream, so it doesn't have to
 * be handled explicitely.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.symmetric
{
	import flash.utils.ByteArray;
	import com.hurlant.util.Memory;
	
	public class SimpleIVMode implements IMode, ICipher
	{
		protected var mode:IVMode;
		protected var cipher:ICipher;
		
		public function SimpleIVMode(mode:IVMode) {
			this.mode = mode;
			cipher = mode as ICipher;
		}
		
		public function getBlockSize():uint {
			return mode.getBlockSize();
		}
		
		public function dispose():void {
			mode.dispose();
			mode = null;
			cipher = null;
			Memory.gc();
		}
		
		public function encrypt(src:ByteArray):void {
			cipher.encrypt(src);
			var tmp:ByteArray = new ByteArray;
			tmp.writeBytes(mode.IV);
			tmp.writeBytes(src);
			src.position=0;
			src.writeBytes(tmp);
		}
		
		public function decrypt(src:ByteArray):void {
			var tmp:ByteArray = new ByteArray;
			tmp.writeBytes(src, 0, getBlockSize());
			mode.IV = tmp;
			tmp = new ByteArray;
			tmp.writeBytes(src, getBlockSize());
			cipher.decrypt(tmp);
			src.length=0;
			src.writeBytes(tmp);
		}
		public function toString():String {
			return "simple-"+cipher.toString();
		}
	}
}