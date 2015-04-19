/**
 * MD5
 * 
 * An ActionScript 3 implementation of the RSA Data Security, Inc. MD5 Message
 * Digest Algorithm, as defined in RFC 1321.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * Derived from
 * 		A JavaScript implementation of the same.
 *		Version 2.1 Copyright (C) Paul Johnston 1999 - 2002.
 * 		Other contributors: Greg Holt, Andrew Kepert, Ydnar, Lostinet
 * 
 * Note:
 * This algorithm should not be your first choice for new developements, but is
 * included to allow interoperability with existing codes and protocols.
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.hash
{
	import flash.utils.ByteArray;
	import flash.utils.Endian;

 	public class MD5 implements IHash
	{
		public static const HASH_SIZE:int = 16;
		public var pad_size:int = 48;

        public function MD5() { }
		
		public function getInputSize():uint
		{
			return 64;
		}
		
		public function getHashSize():uint
		{
			return HASH_SIZE;
		}
		
		public function getPadSize():int 
		{
			return pad_size;
		}
		
		public function hash(src:ByteArray):ByteArray
		{
			var len:uint = src.length *8;
			var savedEndian:String = src.endian;
			// pad to nearest int.
			while (src.length%4!=0) {
				src[src.length]=0;
			}
			// convert ByteArray to an array of uint
			src.position=0;
			var a:Array = [];
			src.endian=Endian.LITTLE_ENDIAN
			for (var i:uint=0;i<src.length;i+=4) {
				a.push(src.readUnsignedInt());
			}
			var h:Array = core_md5(a, len);
			var out:ByteArray = new ByteArray;
			out.endian=Endian.LITTLE_ENDIAN;
			for (i=0;i<4;i++) {
				out.writeUnsignedInt(h[i]);
			}
			// restore length!
			src.length = len/8;
			src.endian = savedEndian;
			
			return out;
		}
		
		private function core_md5(x:Array, len:uint):Array {
		  /* append padding */
		  x[len >> 5] |= 0x80 << ((len) % 32);
		  x[(((len + 64) >>> 9) << 4) + 14] = len;

		  var a:uint = 0x67452301; // 1732584193;
		  var b:uint = 0xEFCDAB89; //-271733879;
		  var c:uint = 0x98BADCFE; //-1732584194;
		  var d:uint = 0x10325476; // 271733878;

		  for(var i:uint = 0; i < x.length; i += 16)
		  {
		  	x[i]||=0;    x[i+1]||=0;  x[i+2]||=0;  x[i+3]||=0;
		  	x[i+4]||=0;  x[i+5]||=0;  x[i+6]||=0;  x[i+7]||=0;
		  	x[i+8]||=0;  x[i+9]||=0;  x[i+10]||=0; x[i+11]||=0;
		  	x[i+12]||=0; x[i+13]||=0; x[i+14]||=0; x[i+15]||=0;

		    var olda:uint = a;
		    var oldb:uint = b;
		    var oldc:uint = c;
		    var oldd:uint = d;
		    
		    a = ff(a, b, c, d, x[i+ 0], 7 , 0xD76AA478);
		    d = ff(d, a, b, c, x[i+ 1], 12, 0xE8C7B756);
		    c = ff(c, d, a, b, x[i+ 2], 17, 0x242070DB);
		    b = ff(b, c, d, a, x[i+ 3], 22, 0xC1BDCEEE);
		    a = ff(a, b, c, d, x[i+ 4], 7 , 0xF57C0FAF);
		    d = ff(d, a, b, c, x[i+ 5], 12, 0x4787C62A);
		    c = ff(c, d, a, b, x[i+ 6], 17, 0xA8304613);
		    b = ff(b, c, d, a, x[i+ 7], 22, 0xFD469501);
		    a = ff(a, b, c, d, x[i+ 8], 7 , 0x698098D8);
		    d = ff(d, a, b, c, x[i+ 9], 12, 0x8B44F7AF);
		    c = ff(c, d, a, b, x[i+10], 17, 0xFFFF5BB1);
		    b = ff(b, c, d, a, x[i+11], 22, 0x895CD7BE);
		    a = ff(a, b, c, d, x[i+12], 7 , 0x6B901122);
		    d = ff(d, a, b, c, x[i+13], 12, 0xFD987193);
		    c = ff(c, d, a, b, x[i+14], 17, 0xA679438E);
		    b = ff(b, c, d, a, x[i+15], 22, 0x49B40821);

		    a = gg(a, b, c, d, x[i+ 1], 5 , 0xf61e2562);
		    d = gg(d, a, b, c, x[i+ 6], 9 , 0xc040b340);
		    c = gg(c, d, a, b, x[i+11], 14, 0x265e5a51);
		    b = gg(b, c, d, a, x[i+ 0], 20, 0xe9b6c7aa);
		    a = gg(a, b, c, d, x[i+ 5], 5 , 0xd62f105d);
		    d = gg(d, a, b, c, x[i+10], 9 ,  0x2441453);
		    c = gg(c, d, a, b, x[i+15], 14, 0xd8a1e681);
		    b = gg(b, c, d, a, x[i+ 4], 20, 0xe7d3fbc8);
		    a = gg(a, b, c, d, x[i+ 9], 5 , 0x21e1cde6);
		    d = gg(d, a, b, c, x[i+14], 9 , 0xc33707d6);
		    c = gg(c, d, a, b, x[i+ 3], 14, 0xf4d50d87);
		    b = gg(b, c, d, a, x[i+ 8], 20, 0x455a14ed);
		    a = gg(a, b, c, d, x[i+13], 5 , 0xa9e3e905);
		    d = gg(d, a, b, c, x[i+ 2], 9 , 0xfcefa3f8);
		    c = gg(c, d, a, b, x[i+ 7], 14, 0x676f02d9);
		    b = gg(b, c, d, a, x[i+12], 20, 0x8d2a4c8a);

		    a = hh(a, b, c, d, x[i+ 5], 4 , 0xfffa3942);
		    d = hh(d, a, b, c, x[i+ 8], 11, 0x8771f681);
		    c = hh(c, d, a, b, x[i+11], 16, 0x6d9d6122);
		    b = hh(b, c, d, a, x[i+14], 23, 0xfde5380c);
		    a = hh(a, b, c, d, x[i+ 1], 4 , 0xa4beea44);
		    d = hh(d, a, b, c, x[i+ 4], 11, 0x4bdecfa9);
		    c = hh(c, d, a, b, x[i+ 7], 16, 0xf6bb4b60);
		    b = hh(b, c, d, a, x[i+10], 23, 0xbebfbc70);
		    a = hh(a, b, c, d, x[i+13], 4 , 0x289b7ec6);
		    d = hh(d, a, b, c, x[i+ 0], 11, 0xeaa127fa);
		    c = hh(c, d, a, b, x[i+ 3], 16, 0xd4ef3085);
		    b = hh(b, c, d, a, x[i+ 6], 23,  0x4881d05);
		    a = hh(a, b, c, d, x[i+ 9], 4 , 0xd9d4d039);
		    d = hh(d, a, b, c, x[i+12], 11, 0xe6db99e5);
		    c = hh(c, d, a, b, x[i+15], 16, 0x1fa27cf8);
		    b = hh(b, c, d, a, x[i+ 2], 23, 0xc4ac5665);
		
		    a = ii(a, b, c, d, x[i+ 0], 6 , 0xf4292244);
		    d = ii(d, a, b, c, x[i+ 7], 10, 0x432aff97);
		    c = ii(c, d, a, b, x[i+14], 15, 0xab9423a7);
		    b = ii(b, c, d, a, x[i+ 5], 21, 0xfc93a039);
		    a = ii(a, b, c, d, x[i+12], 6 , 0x655b59c3);
		    d = ii(d, a, b, c, x[i+ 3], 10, 0x8f0ccc92);
		    c = ii(c, d, a, b, x[i+10], 15, 0xffeff47d);
		    b = ii(b, c, d, a, x[i+ 1], 21, 0x85845dd1);
		    a = ii(a, b, c, d, x[i+ 8], 6 , 0x6fa87e4f);
		    d = ii(d, a, b, c, x[i+15], 10, 0xfe2ce6e0);
		    c = ii(c, d, a, b, x[i+ 6], 15, 0xa3014314);
		    b = ii(b, c, d, a, x[i+13], 21, 0x4e0811a1);
		    a = ii(a, b, c, d, x[i+ 4], 6 , 0xf7537e82);
		    d = ii(d, a, b, c, x[i+11], 10, 0xbd3af235);
		    c = ii(c, d, a, b, x[i+ 2], 15, 0x2ad7d2bb);
		    b = ii(b, c, d, a, x[i+ 9], 21, 0xeb86d391);
		
			a += olda;
			b += oldb;
			c += oldc;
			d += oldd;
			
		  }
		  return [ a, b, c, d ];
		}

		/*
		 * Bitwise rotate a 32-bit number to the left.
		 */
		private function rol(num:uint, cnt:uint):uint
		{
		  return (num << cnt) | (num >>> (32 - cnt));
		}

		/*
		 * These functions implement the four basic operations the algorithm uses.
		 */
		private function cmn(q:uint, a:uint, b:uint, x:uint, s:uint, t:uint):uint {
		  return rol(a + q + x + t, s) + b;
		}
		private function ff(a:uint, b:uint, c:uint, d:uint, x:uint, s:uint, t:uint):uint {
		  return cmn((b & c) | ((~b) & d), a, b, x, s, t);
		}
		private function gg(a:uint, b:uint, c:uint, d:uint, x:uint, s:uint, t:uint):uint {
		  return cmn((b & d) | (c & (~d)), a, b, x, s, t);
		}
		private function hh(a:uint, b:uint, c:uint, d:uint, x:uint, s:uint, t:uint):uint {
		  return cmn(b ^ c ^ d, a, b, x, s, t);
		}
		private function ii(a:uint, b:uint, c:uint, d:uint, x:uint, s:uint, t:uint):uint {
		  return cmn(c ^ (b | (~d)), a, b, x, s, t);
		}

		public function toString():String {
			return "md5";
		}
	}
}
