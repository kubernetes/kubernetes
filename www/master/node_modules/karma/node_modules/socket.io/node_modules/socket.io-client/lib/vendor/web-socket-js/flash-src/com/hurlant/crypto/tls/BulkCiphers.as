/**
 * BulkCiphers
 * 
 * An enumeration of bulk ciphers available for TLS, along with their properties,
 * with a few convenience methods to go with it.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tls {
	import com.hurlant.crypto.Crypto;
	import flash.utils.ByteArray;
	import com.hurlant.crypto.symmetric.ICipher;
	import com.hurlant.crypto.symmetric.TLSPad;
	import com.hurlant.crypto.symmetric.SSLPad;
	
	public class BulkCiphers {
		public static const STREAM_CIPHER:uint = 0;
		public static const BLOCK_CIPHER:uint = 1;

		public static const NULL:uint = 0;
		public static const RC4_40:uint = 1;
		public static const RC4_128:uint = 2
		public static const RC2_CBC_40:uint = 3; // XXX I don't have that one.
		public static const DES_CBC:uint = 4;
		public static const DES3_EDE_CBC:uint = 5;
		public static const DES40_CBC:uint = 6;
		public static const IDEA_CBC:uint = 7; // XXX I don't have that one.
		public static const AES_128:uint = 8;
		public static const AES_256:uint = 9;
		
		private static const algos:Array =
		['', 'rc4', 'rc4', '', 'des-cbc', '3des-cbc', 'des-cbc', '', 'aes', 'aes'];
		
		private static var _props:Array;
		
		init();
		private static function init():void {
			_props = [];
			_props[NULL] 			= new BulkCiphers(STREAM_CIPHER,  0,  0,   0,  0,  0);
			_props[RC4_40] 			= new BulkCiphers(STREAM_CIPHER,  5, 16,  40,  0,  0);
			_props[RC4_128]			= new BulkCiphers(STREAM_CIPHER, 16, 16, 128,  0,  0);
			_props[RC2_CBC_40]		= new BulkCiphers( BLOCK_CIPHER,  5, 16,  40,  8,  8);
			_props[DES_CBC]			= new BulkCiphers( BLOCK_CIPHER,  8,  8,  56,  8,  8);
			_props[DES3_EDE_CBC]	= new BulkCiphers( BLOCK_CIPHER, 24, 24, 168,  8,  8);
			_props[DES40_CBC]		= new BulkCiphers( BLOCK_CIPHER,  5,  8,  40,  8,  8);
			_props[IDEA_CBC]		= new BulkCiphers( BLOCK_CIPHER, 16, 16, 128,  8,  8);
			_props[AES_128]			= new BulkCiphers( BLOCK_CIPHER, 16, 16, 128, 16, 16);
			_props[AES_256]			= new BulkCiphers( BLOCK_CIPHER, 32, 32, 256, 16, 16);
		}
	
		private static function getProp(cipher:uint):BulkCiphers {
			var p:BulkCiphers = _props[cipher];
			if (p==null) {
				throw new Error("Unknown bulk cipher "+cipher.toString(16));
			}
			return p;
		}
		public static function getType(cipher:uint):uint {
			return getProp(cipher).type;
		}
		public static function getKeyBytes(cipher:uint):uint {
			return getProp(cipher).keyBytes;
		}
		public static function getExpandedKeyBytes(cipher:uint):uint {
			return getProp(cipher).expandedKeyBytes;
		}
		public static function getEffectiveKeyBits(cipher:uint):uint {
			return getProp(cipher).effectiveKeyBits;
		}
		public static function getIVSize(cipher:uint):uint {
			return getProp(cipher).IVSize;
		}
		public static function getBlockSize(cipher:uint):uint {
			return getProp(cipher).blockSize;
		}
		public static function getCipher(cipher:uint, key:ByteArray, proto:uint):ICipher {
			if (proto == TLSSecurityParameters.PROTOCOL_VERSION) {
				return Crypto.getCipher(algos[cipher], key, new TLSPad);
			} else {
				return Crypto.getCipher(algos[cipher], key, new SSLPad);
			}
		}

	
		private var type:uint;
		private var keyBytes:uint;
		private var expandedKeyBytes:uint;
		private var effectiveKeyBits:uint;
		private var IVSize:uint;
		private var blockSize:uint;
		
		public function BulkCiphers(t:uint, kb:uint, ekb:uint, fkb:uint, ivs:uint, bs:uint) {
			type = t;
			keyBytes = kb;
			expandedKeyBytes = ekb;
			effectiveKeyBits = fkb;
			IVSize = ivs;
			blockSize = bs;
		}
	}
}
