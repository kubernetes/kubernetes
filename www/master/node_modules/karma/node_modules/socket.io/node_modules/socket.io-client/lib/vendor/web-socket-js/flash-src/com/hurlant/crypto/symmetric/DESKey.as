/**
 * DESKey
 * 
 * An Actionscript 3 implementation of the Data Encryption Standard (DES)
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
	import com.hurlant.util.Hex;
	import com.hurlant.util.Memory;

	public class DESKey implements ISymmetricKey
	{
		/**
		 * what follows is mainly taken from "Applied Cryptography", by Bruce
		 * Schneier, however it also bears great resemblance to Richard
		 * Outerbridge's D3DES...
		 */
	
		private static const Df_Key:Array = [ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32,
				0x10, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67 ];
	
		private static const bytebit:Array = [ 128, 64, 32, 16, 8, 4, 2, 1 ];
	
		private static const bigbyte:Array = [ 0x800000, 0x400000, 0x200000, 0x100000, 0x80000, 0x40000, 0x20000, 0x10000, 0x8000,
				0x4000, 0x2000, 0x1000, 0x800, 0x400, 0x200, 0x100, 0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1 ];
	
		/*
		 * Use the key schedule specified in the Standard (ANSI X3.92-1981).
		 */
	
		private static const pc1:Array = [ 56, 48, 40, 32, 24, 16, 8, 0, 57, 49, 41, 33, 25, 17, 9, 1, 58, 50, 42, 34, 26, 18, 10, 2,
				59, 51, 43, 35, 62, 54, 46, 38, 30, 22, 14, 6, 61, 53, 45, 37, 29, 21, 13, 5, 60, 52, 44, 36, 28, 20, 12,
				4, 27, 19, 11, 3 ];
	
		private static const totrot:Array = [ 1, 2, 4, 6, 8, 10, 12, 14, 15, 17, 19, 21, 23, 25, 27, 28 ];
	
		private static const pc2:Array = [ 13, 16, 10, 23, 0, 4, 2, 27, 14, 5, 20, 9, 22, 18, 11, 3, 25, 7, 15, 6, 26, 19, 12, 1, 40,
				51, 30, 36, 46, 54, 29, 39, 50, 44, 32, 47, 43, 48, 38, 55, 33, 52, 45, 41, 49, 35, 28, 31 ];
	
		private static const SP1:Array = [ 0x01010400, 0x00000000, 0x00010000, 0x01010404, 0x01010004, 0x00010404, 0x00000004,
				0x00010000, 0x00000400, 0x01010400, 0x01010404, 0x00000400, 0x01000404, 0x01010004, 0x01000000, 0x00000004,
				0x00000404, 0x01000400, 0x01000400, 0x00010400, 0x00010400, 0x01010000, 0x01010000, 0x01000404, 0x00010004,
				0x01000004, 0x01000004, 0x00010004, 0x00000000, 0x00000404, 0x00010404, 0x01000000, 0x00010000, 0x01010404,
				0x00000004, 0x01010000, 0x01010400, 0x01000000, 0x01000000, 0x00000400, 0x01010004, 0x00010000, 0x00010400,
				0x01000004, 0x00000400, 0x00000004, 0x01000404, 0x00010404, 0x01010404, 0x00010004, 0x01010000, 0x01000404,
				0x01000004, 0x00000404, 0x00010404, 0x01010400, 0x00000404, 0x01000400, 0x01000400, 0x00000000, 0x00010004,
				0x00010400, 0x00000000, 0x01010004 ];
	
		private static const SP2:Array = [ 0x80108020, 0x80008000, 0x00008000, 0x00108020, 0x00100000, 0x00000020, 0x80100020,
				0x80008020, 0x80000020, 0x80108020, 0x80108000, 0x80000000, 0x80008000, 0x00100000, 0x00000020, 0x80100020,
				0x00108000, 0x00100020, 0x80008020, 0x00000000, 0x80000000, 0x00008000, 0x00108020, 0x80100000, 0x00100020,
				0x80000020, 0x00000000, 0x00108000, 0x00008020, 0x80108000, 0x80100000, 0x00008020, 0x00000000, 0x00108020,
				0x80100020, 0x00100000, 0x80008020, 0x80100000, 0x80108000, 0x00008000, 0x80100000, 0x80008000, 0x00000020,
				0x80108020, 0x00108020, 0x00000020, 0x00008000, 0x80000000, 0x00008020, 0x80108000, 0x00100000, 0x80000020,
				0x00100020, 0x80008020, 0x80000020, 0x00100020, 0x00108000, 0x00000000, 0x80008000, 0x00008020, 0x80000000,
				0x80100020, 0x80108020, 0x00108000 ];
	
		private static const SP3:Array = [ 0x00000208, 0x08020200, 0x00000000, 0x08020008, 0x08000200, 0x00000000, 0x00020208,
				0x08000200, 0x00020008, 0x08000008, 0x08000008, 0x00020000, 0x08020208, 0x00020008, 0x08020000, 0x00000208,
				0x08000000, 0x00000008, 0x08020200, 0x00000200, 0x00020200, 0x08020000, 0x08020008, 0x00020208, 0x08000208,
				0x00020200, 0x00020000, 0x08000208, 0x00000008, 0x08020208, 0x00000200, 0x08000000, 0x08020200, 0x08000000,
				0x00020008, 0x00000208, 0x00020000, 0x08020200, 0x08000200, 0x00000000, 0x00000200, 0x00020008, 0x08020208,
				0x08000200, 0x08000008, 0x00000200, 0x00000000, 0x08020008, 0x08000208, 0x00020000, 0x08000000, 0x08020208,
				0x00000008, 0x00020208, 0x00020200, 0x08000008, 0x08020000, 0x08000208, 0x00000208, 0x08020000, 0x00020208,
				0x00000008, 0x08020008, 0x00020200 ];
	
		private static const SP4:Array = [ 0x00802001, 0x00002081, 0x00002081, 0x00000080, 0x00802080, 0x00800081, 0x00800001,
				0x00002001, 0x00000000, 0x00802000, 0x00802000, 0x00802081, 0x00000081, 0x00000000, 0x00800080, 0x00800001,
				0x00000001, 0x00002000, 0x00800000, 0x00802001, 0x00000080, 0x00800000, 0x00002001, 0x00002080, 0x00800081,
				0x00000001, 0x00002080, 0x00800080, 0x00002000, 0x00802080, 0x00802081, 0x00000081, 0x00800080, 0x00800001,
				0x00802000, 0x00802081, 0x00000081, 0x00000000, 0x00000000, 0x00802000, 0x00002080, 0x00800080, 0x00800081,
				0x00000001, 0x00802001, 0x00002081, 0x00002081, 0x00000080, 0x00802081, 0x00000081, 0x00000001, 0x00002000,
				0x00800001, 0x00002001, 0x00802080, 0x00800081, 0x00002001, 0x00002080, 0x00800000, 0x00802001, 0x00000080,
				0x00800000, 0x00002000, 0x00802080 ];
	
		private static const SP5:Array = [ 0x00000100, 0x02080100, 0x02080000, 0x42000100, 0x00080000, 0x00000100, 0x40000000,
				0x02080000, 0x40080100, 0x00080000, 0x02000100, 0x40080100, 0x42000100, 0x42080000, 0x00080100, 0x40000000,
				0x02000000, 0x40080000, 0x40080000, 0x00000000, 0x40000100, 0x42080100, 0x42080100, 0x02000100, 0x42080000,
				0x40000100, 0x00000000, 0x42000000, 0x02080100, 0x02000000, 0x42000000, 0x00080100, 0x00080000, 0x42000100,
				0x00000100, 0x02000000, 0x40000000, 0x02080000, 0x42000100, 0x40080100, 0x02000100, 0x40000000, 0x42080000,
				0x02080100, 0x40080100, 0x00000100, 0x02000000, 0x42080000, 0x42080100, 0x00080100, 0x42000000, 0x42080100,
				0x02080000, 0x00000000, 0x40080000, 0x42000000, 0x00080100, 0x02000100, 0x40000100, 0x00080000, 0x00000000,
				0x40080000, 0x02080100, 0x40000100 ];
	
		private static const SP6:Array = [ 0x20000010, 0x20400000, 0x00004000, 0x20404010, 0x20400000, 0x00000010, 0x20404010,
				0x00400000, 0x20004000, 0x00404010, 0x00400000, 0x20000010, 0x00400010, 0x20004000, 0x20000000, 0x00004010,
				0x00000000, 0x00400010, 0x20004010, 0x00004000, 0x00404000, 0x20004010, 0x00000010, 0x20400010, 0x20400010,
				0x00000000, 0x00404010, 0x20404000, 0x00004010, 0x00404000, 0x20404000, 0x20000000, 0x20004000, 0x00000010,
				0x20400010, 0x00404000, 0x20404010, 0x00400000, 0x00004010, 0x20000010, 0x00400000, 0x20004000, 0x20000000,
				0x00004010, 0x20000010, 0x20404010, 0x00404000, 0x20400000, 0x00404010, 0x20404000, 0x00000000, 0x20400010,
				0x00000010, 0x00004000, 0x20400000, 0x00404010, 0x00004000, 0x00400010, 0x20004010, 0x00000000, 0x20404000,
				0x20000000, 0x00400010, 0x20004010 ];
	
		private static const SP7:Array = [ 0x00200000, 0x04200002, 0x04000802, 0x00000000, 0x00000800, 0x04000802, 0x00200802,
				0x04200800, 0x04200802, 0x00200000, 0x00000000, 0x04000002, 0x00000002, 0x04000000, 0x04200002, 0x00000802,
				0x04000800, 0x00200802, 0x00200002, 0x04000800, 0x04000002, 0x04200000, 0x04200800, 0x00200002, 0x04200000,
				0x00000800, 0x00000802, 0x04200802, 0x00200800, 0x00000002, 0x04000000, 0x00200800, 0x04000000, 0x00200800,
				0x00200000, 0x04000802, 0x04000802, 0x04200002, 0x04200002, 0x00000002, 0x00200002, 0x04000000, 0x04000800,
				0x00200000, 0x04200800, 0x00000802, 0x00200802, 0x04200800, 0x00000802, 0x04000002, 0x04200802, 0x04200000,
				0x00200800, 0x00000000, 0x00000002, 0x04200802, 0x00000000, 0x00200802, 0x04200000, 0x00000800, 0x04000002,
				0x04000800, 0x00000800, 0x00200002 ];
	
		private static const SP8:Array = [ 0x10001040, 0x00001000, 0x00040000, 0x10041040, 0x10000000, 0x10001040, 0x00000040,
				0x10000000, 0x00040040, 0x10040000, 0x10041040, 0x00041000, 0x10041000, 0x00041040, 0x00001000, 0x00000040,
				0x10040000, 0x10000040, 0x10001000, 0x00001040, 0x00041000, 0x00040040, 0x10040040, 0x10041000, 0x00001040,
				0x00000000, 0x00000000, 0x10040040, 0x10000040, 0x10001000, 0x00041040, 0x00040000, 0x00041040, 0x00040000,
				0x10041000, 0x00001000, 0x00000040, 0x10040040, 0x00001000, 0x00041040, 0x10001000, 0x00000040, 0x10000040,
				0x10040000, 0x10040040, 0x10000000, 0x00040000, 0x10001040, 0x00000000, 0x10041040, 0x00040040, 0x10000040,
				0x10040000, 0x10001000, 0x10001040, 0x00000000, 0x10041040, 0x00041000, 0x00041000, 0x00001040, 0x00001040,
				0x00040040, 0x10000000, 0x10041000 ];

	
		protected var key:ByteArray;
		protected var encKey:Array;
		protected var decKey:Array;
	

		public function DESKey(key:ByteArray) {
			this.key = key;
			this.encKey = generateWorkingKey(true, key, 0);
			this.decKey = generateWorkingKey(false, key, 0);
		}

		public function getBlockSize():uint
		{
			return 8;
		}
		
		public function decrypt(block:ByteArray, index:uint=0):void
		{
			desFunc(decKey, block, index, block, index);
		}
		
		public function dispose():void
		{
			var i:uint=0;
			for (i=0;i<encKey.length;i++) { encKey[i]=0; }
			for (i=0;i<decKey.length;i++) { decKey[i]=0; }
			encKey=null;
			decKey=null;
			for (i=0;i<key.length;i++) { key[i]=0; }
			key.length = 0;
			key = null;
			Memory.gc();
		}
		
		public function encrypt(block:ByteArray, index:uint=0):void
		{
			desFunc(encKey, block, index, block, index);
		}
			
	
		/**
		 * generate an integer based working key based on our secret key and what we
		 * processing we are planning to do.
		 * 
		 * Acknowledgements for this routine go to James Gillogly & Phil Karn.
		 */
		protected function generateWorkingKey(encrypting:Boolean, key:ByteArray, off:uint):Array
		{
			//int[] newKey = new int[32];
			var newKey:Array = [];
			//boolean[] pc1m = new boolean[56], pcr = new boolean[56];
			var pc1m:ByteArray = new ByteArray;
			var pcr:ByteArray = new ByteArray;
			
			var l:uint;
	
			for (var j:uint = 0; j < 56; j++)
			{
				l = pc1[j];
	
				pc1m[j] = ((key[off + (l >>> 3)] & bytebit[l & 07]) != 0);
			}
	
			for (var i:uint = 0; i < 16; i++)
			{
				var m:uint;
				var n:uint;
	
				if (encrypting)
				{
					m = i << 1;
				}
				else
				{
					m = (15 - i) << 1;
				}
	
				n = m + 1;
				newKey[m] = newKey[n] = 0;
	
				for (j = 0; j < 28; j++)
				{
					l = j + totrot[i];
					if (l < 28)
					{
						pcr[j] = pc1m[l];
					}
					else
					{
						pcr[j] = pc1m[l - 28];
					}
				}
	
				for (j = 28; j < 56; j++)
				{
					l = j + totrot[i];
					if (l < 56)
					{
						pcr[j] = pc1m[l];
					}
					else
					{
						pcr[j] = pc1m[l - 28];
					}
				}
	
				for (j = 0; j < 24; j++)
				{
					if (pcr[pc2[j]])
					{
						newKey[m] |= bigbyte[j];
					}
	
					if (pcr[pc2[j + 24]])
					{
						newKey[n] |= bigbyte[j];
					}
				}
			}
	
			//
			// store the processed key
			//
			for (i = 0; i != 32; i += 2)
			{
				var i1:uint;
				var i2:uint;
	
				i1 = newKey[i];
				i2 = newKey[i + 1];
	
				newKey[i] = ((i1 & 0x00fc0000) << 6) | ((i1 & 0x00000fc0) << 10) | ((i2 & 0x00fc0000) >>> 10)
						| ((i2 & 0x00000fc0) >>> 6);
	
				newKey[i + 1] = ((i1 & 0x0003f000) << 12) | ((i1 & 0x0000003f) << 16) | ((i2 & 0x0003f000) >>> 4)
						| (i2 & 0x0000003f);
			}
			return newKey;
		}
	
		/**
		 * the DES engine.
		 */
		protected function desFunc(wKey:Array, inp:ByteArray, inOff:uint, out:ByteArray, outOff:uint):void
		{
			var work:uint;
			var right:uint;
			var left:uint;
	
			left = (inp[inOff + 0] & 0xff) << 24;
			left |= (inp[inOff + 1] & 0xff) << 16;
			left |= (inp[inOff + 2] & 0xff) << 8;
			left |= (inp[inOff + 3] & 0xff);
	
			right = (inp[inOff + 4] & 0xff) << 24;
			right |= (inp[inOff + 5] & 0xff) << 16;
			right |= (inp[inOff + 6] & 0xff) << 8;
			right |= (inp[inOff + 7] & 0xff);

			work = ((left >>> 4) ^ right) & 0x0f0f0f0f;
			right ^= work;
			left ^= (work << 4);
			work = ((left >>> 16) ^ right) & 0x0000ffff;
			right ^= work;
			left ^= (work << 16);
			work = ((right >>> 2) ^ left) & 0x33333333;
			left ^= work;
			right ^= (work << 2);
			work = ((right >>> 8) ^ left) & 0x00ff00ff;
			left ^= work;
			right ^= (work << 8);
			right = ((right << 1) | ((right >>> 31) & 1)) & 0xffffffff;
			work = (left ^ right) & 0xaaaaaaaa;
			left ^= work;
			right ^= work;
			left = ((left << 1) | ((left >>> 31) & 1)) & 0xffffffff;

			for (var round:uint = 0; round < 8; round++)
			{
				var fval:uint;
	
				work = (right << 28) | (right >>> 4);
				work ^= wKey[round * 4 + 0];
				fval = SP7[work & 0x3f];
				fval |= SP5[(work >>> 8) & 0x3f];
				fval |= SP3[(work >>> 16) & 0x3f];
				fval |= SP1[(work >>> 24) & 0x3f];
				work = right ^ wKey[round * 4 + 1];
				fval |= SP8[work & 0x3f];
				fval |= SP6[(work >>> 8) & 0x3f];
				fval |= SP4[(work >>> 16) & 0x3f];
				fval |= SP2[(work >>> 24) & 0x3f];
				left ^= fval;
				work = (left << 28) | (left >>> 4);
				work ^= wKey[round * 4 + 2];
				fval = SP7[work & 0x3f];
				fval |= SP5[(work >>> 8) & 0x3f];
				fval |= SP3[(work >>> 16) & 0x3f];
				fval |= SP1[(work >>> 24) & 0x3f];
				work = left ^ wKey[round * 4 + 3];
				fval |= SP8[work & 0x3f];
				fval |= SP6[(work >>> 8) & 0x3f];
				fval |= SP4[(work >>> 16) & 0x3f];
				fval |= SP2[(work >>> 24) & 0x3f];
				right ^= fval;
			}
	
			right = (right << 31) | (right >>> 1);
			work = (left ^ right) & 0xaaaaaaaa;
			left ^= work;
			right ^= work;
			left = (left << 31) | (left >>> 1);
			work = ((left >>> 8) ^ right) & 0x00ff00ff;
			right ^= work;
			left ^= (work << 8);
			work = ((left >>> 2) ^ right) & 0x33333333;
			right ^= work;
			left ^= (work << 2);
			work = ((right >>> 16) ^ left) & 0x0000ffff;
			left ^= work;
			right ^= (work << 16);
			work = ((right >>> 4) ^ left) & 0x0f0f0f0f;
			left ^= work;
			right ^= (work << 4);
	
			out[outOff + 0] = ((right >>> 24) & 0xff);
			out[outOff + 1] = ((right >>> 16) & 0xff);
			out[outOff + 2] = ((right >>> 8) & 0xff);
			out[outOff + 3] = (right & 0xff);
			out[outOff + 4] = ((left >>> 24) & 0xff);
			out[outOff + 5] = ((left >>> 16) & 0xff);
			out[outOff + 6] = ((left >>> 8) & 0xff);
			out[outOff + 7] = (left & 0xff);
		}
		
		
		public function toString():String {
			return "des";
		}
		
		
		
	}
}