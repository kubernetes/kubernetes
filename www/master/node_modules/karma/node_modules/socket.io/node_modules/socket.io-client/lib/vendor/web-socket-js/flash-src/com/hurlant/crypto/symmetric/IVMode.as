/**
 * IVMode
 * 
 * An abstract class for confidentialy modes that rely on an initialization vector.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.symmetric
{
	import com.hurlant.crypto.prng.Random;
	import com.hurlant.crypto.tests.TestCase;
	import com.hurlant.util.Memory;
	
	import flash.utils.ByteArray;
	
	/**
	 * An "abtract" class to avoid redundant code in subclasses
	 */
	public class IVMode
	{
		protected var key:ISymmetricKey;
		protected var padding:IPad;
		// random generator used to generate IVs
		protected var prng:Random;
		// optional static IV. used for testing only.
		protected var iv:ByteArray;
		// generated IV is stored here.
		protected var lastIV:ByteArray;
		protected var blockSize:uint;
		
		
		public function IVMode(key:ISymmetricKey, padding:IPad = null) {
			this.key = key;
			blockSize = key.getBlockSize();
			if (padding == null) {
				padding = new PKCS5(blockSize);
			} else {
				padding.setBlockSize(blockSize);
			}
			this.padding = padding;
			
			prng = new Random;
			iv = null;
			lastIV = new ByteArray;
		}
		
		public function getBlockSize():uint {
			return key.getBlockSize();
		}
		public function dispose():void {
			var i:uint;
			if (iv != null) {
				for (i=0;i<iv.length;i++) {
					iv[i] = prng.nextByte();
				}
				iv.length=0;
				iv = null;
			}
			if (lastIV != null) {
				for (i=0;i<iv.length;i++) {
					lastIV[i] = prng.nextByte();
				}
				lastIV.length=0;
				lastIV=null;
			}
			key.dispose();
			key = null;
			padding = null;
			prng.dispose();
			prng = null;
			Memory.gc();
		}
		/**
		 * Optional function to force the IV value.
		 * Normally, an IV gets generated randomly at every encrypt() call.
		 * Also, use this to set the IV before calling decrypt()
		 * (if not set before decrypt(), the IV is read from the beginning of the stream.)
		 */
		public function set IV(value:ByteArray):void {
			iv = value;
			lastIV.length=0;
			lastIV.writeBytes(iv);
		}
		public function get IV():ByteArray {
			return lastIV;
		}
		
		protected function getIV4e():ByteArray {
			var vec:ByteArray = new ByteArray;
			if (iv) {
				vec.writeBytes(iv);
			} else {
				prng.nextBytes(vec, blockSize);
			}
			lastIV.length=0;
			lastIV.writeBytes(vec);
			return vec;
		}
		protected function getIV4d():ByteArray {
			var vec:ByteArray = new ByteArray;
			if (iv) {
				vec.writeBytes(iv);
			} else {
				throw new Error("an IV must be set before calling decrypt()");
			}
			return vec;
		}
	}
}