package com.gsolo.encryption {	
	public class MD5 {
		/*
		 * A JavaScript implementation of the RSA Data Security, Inc. MD5 Message
		 * Digest Algorithm, as defined in RFC 1321.
		 * Version 2.2-alpha Copyright (C) Paul Johnston 1999 - 2005
		 * Other contributors: Greg Holt, Andrew Kepert, Ydnar, Lostinet
		 * Distributed under the BSD License
		 * See http://pajhome.org.uk/crypt/md5 for more info.
		 *
		 * Converted to AS3 By Geoffrey Williams
		 */
		
		/*
		 * Configurable variables. You may need to tweak these to be compatible with
		 * the server-side, but the defaults work in most cases.
		 */
		
		public static const HEX_FORMAT_LOWERCASE:uint = 0;
		public static const HEX_FORMAT_UPPERCASE:uint = 1;
		 
		public static const BASE64_PAD_CHARACTER_DEFAULT_COMPLIANCE:String = "";
		public static const BASE64_PAD_CHARACTER_RFC_COMPLIANCE:String = "=";
		 
		public static var hexcase:uint = 0;   /* hex output format. 0 - lowercase; 1 - uppercase        */
		public static var b64pad:String  = ""; /* base-64 pad character. "=" for strict RFC compliance   */
		
		public static function encrypt (string:String):String {
			return hex_md5 (string);
		}
		
		/*
		 * These are the functions you'll usually want to call
		 * They take string arguments and return either hex or base-64 encoded strings
		 */
		public static function hex_md5 (string:String):String {
			return rstr2hex (rstr_md5 (str2rstr_utf8 (string)));
		}
		
		public static function b64_md5 (string:String):String {
			return rstr2b64 (rstr_md5 (str2rstr_utf8 (string)));
		}
		
		public static function any_md5 (string:String, encoding:String):String {
			return rstr2any (rstr_md5 (str2rstr_utf8 (string)), encoding);
		}
		public static function hex_hmac_md5 (key:String, data:String):String {
			return rstr2hex (rstr_hmac_md5 (str2rstr_utf8 (key), str2rstr_utf8 (data)));
		}
		public static function b64_hmac_md5 (key:String, data:String):String {
			return rstr2b64 (rstr_hmac_md5 (str2rstr_utf8 (key), str2rstr_utf8 (data)));
		}
		public static function any_hmac_md5 (key:String, data:String, encoding:String):String {
			return rstr2any(rstr_hmac_md5(str2rstr_utf8(key), str2rstr_utf8(data)), encoding);
		}
		
		/*
		 * Perform a simple self-test to see if the VM is working
		 */
		public static function md5_vm_test ():Boolean {
		  return hex_md5 ("abc") == "900150983cd24fb0d6963f7d28e17f72";
		}
		
		/*
		 * Calculate the MD5 of a raw string
		 */
		public static function rstr_md5 (string:String):String {
		  return binl2rstr (binl_md5 (rstr2binl (string), string.length * 8));
		}
		
		/*
		 * Calculate the HMAC-MD5, of a key and some data (raw strings)
		 */
		public static function rstr_hmac_md5 (key:String, data:String):String {
		  var bkey:Array = rstr2binl (key);
		  if (bkey.length > 16) bkey = binl_md5 (bkey, key.length * 8);
		
		  var ipad:Array = new Array(16), opad:Array = new Array(16);
		  for(var i:Number = 0; i < 16; i++) {
		    ipad[i] = bkey[i] ^ 0x36363636;
		    opad[i] = bkey[i] ^ 0x5C5C5C5C;
		  }
		
		  var hash:Array = binl_md5 (ipad.concat (rstr2binl (data)), 512 + data.length * 8);
		  return binl2rstr (binl_md5 (opad.concat (hash), 512 + 128));
		}
		
		/*
		 * Convert a raw string to a hex string
		 */
		public static function rstr2hex (input:String):String {
		  var hex_tab:String = hexcase ? "0123456789ABCDEF" : "0123456789abcdef";
		  var output:String = "";
		  var x:Number;
		  for(var i:Number = 0; i < input.length; i++) {
		  	x = input.charCodeAt(i);
		    output += hex_tab.charAt((x >>> 4) & 0x0F)
		           +  hex_tab.charAt( x        & 0x0F);
		  }
		  return output;
		}
		
		/*
		 * Convert a raw string to a base-64 string
		 */
		public static function rstr2b64 (input:String):String {
		  var tab:String = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
		  var output:String = "";
		  var len:Number = input.length;
		  for(var i:Number = 0; i < len; i += 3) {
		    var triplet:Number = (input.charCodeAt(i) << 16)
		                | (i + 1 < len ? input.charCodeAt(i+1) << 8 : 0)
		                | (i + 2 < len ? input.charCodeAt(i+2)      : 0);
		    for(var j:Number = 0; j < 4; j++) {
		      if(i * 8 + j * 6 > input.length * 8) output += b64pad;
		      else output += tab.charAt((triplet >>> 6*(3-j)) & 0x3F);
		    }
		  }
		  return output;
		}
		
		/*
		 * Convert a raw string to an arbitrary string encoding
		 */
		public static function rstr2any(input:String, encoding:String):String {
		  var divisor:Number = encoding.length;
		  var remainders:Array = [];
		  var i:Number, q:Number, x:Number, quotient:Array;
		
		  /* Convert to an array of 16-bit big-endian values, forming the dividend */
		  var dividend:Array = new Array(input.length / 2);
		  for(i = 0; i < dividend.length; i++) {
		    dividend[i] = (input.charCodeAt(i * 2) << 8) | input.charCodeAt(i * 2 + 1);
		  }
		
		  /*
		   * Repeatedly perform a long division. The binary array forms the dividend,
		   * the length of the encoding is the divisor. Once computed, the quotient
		   * forms the dividend for the next step. We stop when the dividend is zero.
		   * All remainders are stored for later use.
		   */
		  while(dividend.length > 0) {
		    quotient = [];
		    x = 0;
		    for(i = 0; i < dividend.length; i++) {
		      x = (x << 16) + dividend[i];
		      q = Math.floor(x / divisor);
		      x -= q * divisor;
		      if(quotient.length > 0 || q > 0)
		        quotient[quotient.length] = q;
		    }
		    remainders[remainders.length] = x;
		    dividend = quotient;
		  }
		
		  /* Convert the remainders to the output string */
		  var output:String = "";
		  for(i = remainders.length - 1; i >= 0; i--)
		    output += encoding.charAt (remainders[i]);
		
		  return output;
		}
		
		/*
		 * Encode a string as utf-8.
		 * For efficiency, this assumes the input is valid utf-16.
		 */
		public static function str2rstr_utf8 (input:String):String {
		  var output:String = "";
		  var i:Number = -1;
		  var x:Number, y:Number;
		
		  while(++i < input.length) {
		    /* Decode utf-16 surrogate pairs */
		    x = input.charCodeAt(i);
		    y = i + 1 < input.length ? input.charCodeAt(i + 1) : 0;
		    if(0xD800 <= x && x <= 0xDBFF && 0xDC00 <= y && y <= 0xDFFF) {
		      x = 0x10000 + ((x & 0x03FF) << 10) + (y & 0x03FF);
		      i++;
		    }
		
		    /* Encode output as utf-8 */
		    if(x <= 0x7F)
		      output += String.fromCharCode(x);
		    else if(x <= 0x7FF)
		      output += String.fromCharCode(0xC0 | ((x >>> 6 ) & 0x1F),
		                                    0x80 | ( x         & 0x3F));
		    else if(x <= 0xFFFF)
		      output += String.fromCharCode(0xE0 | ((x >>> 12) & 0x0F),
		                                    0x80 | ((x >>> 6 ) & 0x3F),
		                                    0x80 | ( x         & 0x3F));
		    else if(x <= 0x1FFFFF)
		      output += String.fromCharCode(0xF0 | ((x >>> 18) & 0x07),
		                                    0x80 | ((x >>> 12) & 0x3F),
		                                    0x80 | ((x >>> 6 ) & 0x3F),
		                                    0x80 | ( x         & 0x3F));
		  }
		  return output;
		}
		
		/*
		 * Encode a string as utf-16
		 */
		public static function str2rstr_utf16le (input:String):String {
		  var output:String = "";
		  for(var i:Number = 0; i < input.length; i++)
		    output += String.fromCharCode( input.charCodeAt(i)        & 0xFF,
		                                  (input.charCodeAt(i) >>> 8) & 0xFF);
		  return output;
		}
		
		public static function str2rstr_utf16be (input:String):String {
		  var output:String = "";
		  for(var i:Number = 0; i < input.length; i++)
		    output += String.fromCharCode((input.charCodeAt(i) >>> 8) & 0xFF,
		                                   input.charCodeAt(i)        & 0xFF);
		  return output;
		}
		
		/*
		 * Convert a raw string to an array of little-endian words
		 * Characters >255 have their high-byte silently ignored.
		 */
		public static function rstr2binl (input:String):Array {
		  var output:Array = new Array(input.length >> 2);
		  for(var i:Number = 0; i < output.length; i++)
		    output[i] = 0;
		  for(i = 0; i < input.length * 8; i += 8)
		    output[i>>5] |= (input.charCodeAt(i / 8) & 0xFF) << (i%32);
		  return output;
		}
		
		/*
		 * Convert an array of little-endian words to a string
		 */
		public static function binl2rstr (input:Array):String {
		  var output:String = "";
		  for(var i:Number = 0; i < input.length * 32; i += 8)
		    output += String.fromCharCode((input[i>>5] >>> (i % 32)) & 0xFF);
		  return output;
		}
		
		/*
		 * Calculate the MD5 of an array of little-endian words, and a bit length.
		 */
		public static function binl_md5 (x:Array, len:Number):Array {
		  /* append padding */
		  x[len >> 5] |= 0x80 << ((len) % 32);
		  x[(((len + 64) >>> 9) << 4) + 14] = len;
		
		  var a:Number =  1732584193;
		  var b:Number = -271733879;
		  var c:Number = -1732584194;
		  var d:Number =  271733878;
		
		  for(var i:Number = 0; i < x.length; i += 16) {
		    var olda:Number = a;
		    var oldb:Number = b;
		    var oldc:Number = c;
		    var oldd:Number = d;
		
		    a = md5_ff(a, b, c, d, x[i+ 0], 7 , -680876936);
		    d = md5_ff(d, a, b, c, x[i+ 1], 12, -389564586);
		    c = md5_ff(c, d, a, b, x[i+ 2], 17,  606105819);
		    b = md5_ff(b, c, d, a, x[i+ 3], 22, -1044525330);
		    a = md5_ff(a, b, c, d, x[i+ 4], 7 , -176418897);
		    d = md5_ff(d, a, b, c, x[i+ 5], 12,  1200080426);
		    c = md5_ff(c, d, a, b, x[i+ 6], 17, -1473231341);
		    b = md5_ff(b, c, d, a, x[i+ 7], 22, -45705983);
		    a = md5_ff(a, b, c, d, x[i+ 8], 7 ,  1770035416);
		    d = md5_ff(d, a, b, c, x[i+ 9], 12, -1958414417);
		    c = md5_ff(c, d, a, b, x[i+10], 17, -42063);
		    b = md5_ff(b, c, d, a, x[i+11], 22, -1990404162);
		    a = md5_ff(a, b, c, d, x[i+12], 7 ,  1804603682);
		    d = md5_ff(d, a, b, c, x[i+13], 12, -40341101);
		    c = md5_ff(c, d, a, b, x[i+14], 17, -1502002290);
		    b = md5_ff(b, c, d, a, x[i+15], 22,  1236535329);
		
		    a = md5_gg(a, b, c, d, x[i+ 1], 5 , -165796510);
		    d = md5_gg(d, a, b, c, x[i+ 6], 9 , -1069501632);
		    c = md5_gg(c, d, a, b, x[i+11], 14,  643717713);
		    b = md5_gg(b, c, d, a, x[i+ 0], 20, -373897302);
		    a = md5_gg(a, b, c, d, x[i+ 5], 5 , -701558691);
		    d = md5_gg(d, a, b, c, x[i+10], 9 ,  38016083);
		    c = md5_gg(c, d, a, b, x[i+15], 14, -660478335);
		    b = md5_gg(b, c, d, a, x[i+ 4], 20, -405537848);
		    a = md5_gg(a, b, c, d, x[i+ 9], 5 ,  568446438);
		    d = md5_gg(d, a, b, c, x[i+14], 9 , -1019803690);
		    c = md5_gg(c, d, a, b, x[i+ 3], 14, -187363961);
		    b = md5_gg(b, c, d, a, x[i+ 8], 20,  1163531501);
		    a = md5_gg(a, b, c, d, x[i+13], 5 , -1444681467);
		    d = md5_gg(d, a, b, c, x[i+ 2], 9 , -51403784);
		    c = md5_gg(c, d, a, b, x[i+ 7], 14,  1735328473);
		    b = md5_gg(b, c, d, a, x[i+12], 20, -1926607734);
		
		    a = md5_hh(a, b, c, d, x[i+ 5], 4 , -378558);
		    d = md5_hh(d, a, b, c, x[i+ 8], 11, -2022574463);
		    c = md5_hh(c, d, a, b, x[i+11], 16,  1839030562);
		    b = md5_hh(b, c, d, a, x[i+14], 23, -35309556);
		    a = md5_hh(a, b, c, d, x[i+ 1], 4 , -1530992060);
		    d = md5_hh(d, a, b, c, x[i+ 4], 11,  1272893353);
		    c = md5_hh(c, d, a, b, x[i+ 7], 16, -155497632);
		    b = md5_hh(b, c, d, a, x[i+10], 23, -1094730640);
		    a = md5_hh(a, b, c, d, x[i+13], 4 ,  681279174);
		    d = md5_hh(d, a, b, c, x[i+ 0], 11, -358537222);
		    c = md5_hh(c, d, a, b, x[i+ 3], 16, -722521979);
		    b = md5_hh(b, c, d, a, x[i+ 6], 23,  76029189);
		    a = md5_hh(a, b, c, d, x[i+ 9], 4 , -640364487);
		    d = md5_hh(d, a, b, c, x[i+12], 11, -421815835);
		    c = md5_hh(c, d, a, b, x[i+15], 16,  530742520);
		    b = md5_hh(b, c, d, a, x[i+ 2], 23, -995338651);
		
		    a = md5_ii(a, b, c, d, x[i+ 0], 6 , -198630844);
		    d = md5_ii(d, a, b, c, x[i+ 7], 10,  1126891415);
		    c = md5_ii(c, d, a, b, x[i+14], 15, -1416354905);
		    b = md5_ii(b, c, d, a, x[i+ 5], 21, -57434055);
		    a = md5_ii(a, b, c, d, x[i+12], 6 ,  1700485571);
		    d = md5_ii(d, a, b, c, x[i+ 3], 10, -1894986606);
		    c = md5_ii(c, d, a, b, x[i+10], 15, -1051523);
		    b = md5_ii(b, c, d, a, x[i+ 1], 21, -2054922799);
		    a = md5_ii(a, b, c, d, x[i+ 8], 6 ,  1873313359);
		    d = md5_ii(d, a, b, c, x[i+15], 10, -30611744);
		    c = md5_ii(c, d, a, b, x[i+ 6], 15, -1560198380);
		    b = md5_ii(b, c, d, a, x[i+13], 21,  1309151649);
		    a = md5_ii(a, b, c, d, x[i+ 4], 6 , -145523070);
		    d = md5_ii(d, a, b, c, x[i+11], 10, -1120210379);
		    c = md5_ii(c, d, a, b, x[i+ 2], 15,  718787259);
		    b = md5_ii(b, c, d, a, x[i+ 9], 21, -343485551);
		
		    a = safe_add(a, olda);
		    b = safe_add(b, oldb);
		    c = safe_add(c, oldc);
		    d = safe_add(d, oldd);
		  }
		  return [a, b, c, d];
		}
		
		/*
		 * These functions implement the four basic operations the algorithm uses.
		 */
		public static function md5_cmn (q:Number, a:Number, b:Number, x:Number, s:Number, t:Number):Number {
		  return safe_add (bit_rol (safe_add (safe_add (a, q), safe_add(x, t)), s), b);
		}
		public static function md5_ff (a:Number, b:Number, c:Number, d:Number, x:Number, s:Number, t:Number):Number {
		  return md5_cmn ((b & c) | ((~b) & d), a, b, x, s, t);
		}
		public static function md5_gg (a:Number, b:Number, c:Number, d:Number, x:Number, s:Number, t:Number):Number {
		  return md5_cmn ((b & d) | (c & (~d)), a, b, x, s, t);
		}
		public static function md5_hh (a:Number, b:Number, c:Number, d:Number, x:Number, s:Number, t:Number):Number {
		  return md5_cmn (b ^ c ^ d, a, b, x, s, t);
		}
		public static function md5_ii (a:Number, b:Number, c:Number, d:Number, x:Number, s:Number, t:Number):Number {
		  return md5_cmn (c ^ (b | (~d)), a, b, x, s, t);
		}
		
		/*
		 * Add integers, wrapping at 2^32. This uses 16-bit operations internally
		 * to work around bugs in some JS interpreters.
		 */
		public static function safe_add (x:Number, y:Number):Number {
		  var lsw:Number = (x & 0xFFFF) + (y & 0xFFFF);
		  var msw:Number = (x >> 16) + (y >> 16) + (lsw >> 16);
		  return (msw << 16) | (lsw & 0xFFFF);
		}
		
		/*
		 * Bitwise rotate a 32-bit number to the left.
		 */
		public static function bit_rol (num:Number, cnt:Number):Number {
		  return (num << cnt) | (num >>> (32 - cnt));
		}
				
	}
}
