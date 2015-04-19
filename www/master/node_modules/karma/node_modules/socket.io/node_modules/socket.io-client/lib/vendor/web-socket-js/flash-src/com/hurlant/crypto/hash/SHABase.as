/**
 * SHABase
 * 
 * An ActionScript 3 abstract class for the SHA family of hash functions
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.hash
{
	import flash.utils.ByteArray;
	import flash.utils.Endian;

	public class SHABase implements IHash
	{

        public function SHABase() { }

		public var pad_size:int = 40;
		public function getInputSize():uint
		{
			return 64;
		}
		
		public function getHashSize():uint
		{
			return 0;
		}
		
		public function getPadSize():int 
		{
			return pad_size;
		}
		
		public function hash(src:ByteArray):ByteArray
		{
			var savedLength:uint = src.length;
			var savedEndian:String = src.endian;
			
			src.endian = Endian.BIG_ENDIAN;
			var len:uint = savedLength *8;
			// pad to nearest int.
			while (src.length%4!=0) {
				src[src.length]=0;
			}
			// convert ByteArray to an array of uint
			src.position=0;
			var a:Array = [];
			for (var i:uint=0;i<src.length;i+=4) {
				a.push(src.readUnsignedInt());
			}
			var h:Array = core(a, len);
			var out:ByteArray = new ByteArray;
			var words:uint = getHashSize()/4;
			for (i=0;i<words;i++) {
				out.writeUnsignedInt(h[i]);
			}
			// unpad, to leave the source untouched.
			src.length = savedLength;
			src.endian = savedEndian;
			return out;
		}
		protected function core(x:Array, len:uint):Array {
			return null;
		}
		
		public function toString():String {
			return "sha";
		}
	}
}
