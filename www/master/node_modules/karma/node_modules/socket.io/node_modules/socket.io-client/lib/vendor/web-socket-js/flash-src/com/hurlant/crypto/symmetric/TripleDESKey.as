/**
 * TripleDESKey
 * 
 * An Actionscript 3 implementation of Triple DES
 * Copyright (c) 2007 Henri Torgemane
 * 
 * Derived from:
 * 		The Bouncy Castle Crypto package, 
 * 		Copyright (c) 2000-2004 The Legion Of The Bouncy Castle
 * 		(http://www.bouncycastle.org)
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.symmetric
{
	import flash.utils.ByteArray;
	import com.hurlant.util.Memory;
	import com.hurlant.util.Hex;

	public class TripleDESKey extends DESKey
	{
		protected var encKey2:Array;
		protected var encKey3:Array;
		protected var decKey2:Array;
		protected var decKey3:Array;
		
		/**
		 * This supports 2TDES and 3TDES.
		 * If the key passed is 128 bits, 2TDES is used.
		 * If the key has 192 bits, 3TDES is used.
		 * Other key lengths give "undefined" results.
		 */
		public function TripleDESKey(key:ByteArray)
		{
			super(key);
			encKey2 = generateWorkingKey(false, key, 8);
			decKey2 = generateWorkingKey(true, key, 8);
			if (key.length>16) {
				encKey3 = generateWorkingKey(true, key, 16);
				decKey3 = generateWorkingKey(false, key, 16);
			} else {
				encKey3 = encKey;
				decKey3 = decKey;
			}
		}

		public override function dispose():void
		{
			super.dispose();
			var i:uint = 0;
			if (encKey2!=null) {
				for (i=0;i<encKey2.length;i++) { encKey2[i]=0; }
				encKey2=null;
			}
			if (encKey3!=null) {
				for (i=0;i<encKey3.length;i++) { encKey3[i]=0; }
				encKey3=null;
			}
			if (decKey2!=null) {
				for (i=0;i<decKey2.length;i++) { decKey2[i]=0; }
				decKey2=null
			}
			if (decKey3!=null) {
				for (i=0;i<decKey3.length;i++) { decKey3[i]=0; }
				decKey3=null;
			}
			Memory.gc();
		}
		
		public override function encrypt(block:ByteArray, index:uint=0):void
		{
			desFunc(encKey, block,index, block,index);
			desFunc(encKey2, block,index, block,index);
			desFunc(encKey3, block,index, block,index);
		}
		
		public override function decrypt(block:ByteArray, index:uint=0):void
		{
			desFunc(decKey3, block, index, block, index);
			desFunc(decKey2, block, index, block, index);
			desFunc(decKey, block, index, block, index);
		}
		
		public override function toString():String {
			return "3des";
		}
	}
}