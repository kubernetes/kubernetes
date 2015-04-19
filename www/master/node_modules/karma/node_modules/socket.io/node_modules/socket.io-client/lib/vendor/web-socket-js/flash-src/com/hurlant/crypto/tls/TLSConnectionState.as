/**
 * TLSConnectionState
 * 
 * This class encapsulates the read or write state of a TLS connection,
 * and implementes the encrypting and hashing of packets. 
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tls {
	import flash.utils.IDataInput;
	import flash.utils.ByteArray;
	import com.hurlant.crypto.hash.MD5;
	import com.hurlant.crypto.hash.HMAC;
	import com.hurlant.crypto.hash.IHash;
	import com.hurlant.crypto.symmetric.ICipher;
	import com.hurlant.crypto.symmetric.IVMode;
	import com.hurlant.util.Hex;
	import com.hurlant.util.ArrayUtil;
	
	public class TLSConnectionState implements IConnectionState {


		// compression state
		
		// cipher state
		private var bulkCipher:uint;
		private var cipherType:uint;
		private var CIPHER_key:ByteArray;
		private var CIPHER_IV:ByteArray;
		private var cipher:ICipher;
		private var ivmode:IVMode;
		
		// mac secret
		private var macAlgorithm:uint;
		private var MAC_write_secret:ByteArray;
		private var hmac:HMAC;
		
		// sequence number. uint64
		private var seq_lo:uint;
		private var seq_hi:uint;
		


		public function TLSConnectionState(
				bulkCipher:uint=0, cipherType:uint=0, macAlgorithm:uint=0,
				mac:ByteArray=null, key:ByteArray=null, IV:ByteArray=null) {
			this.bulkCipher = bulkCipher;
			this.cipherType = cipherType;
			this.macAlgorithm = macAlgorithm;
			MAC_write_secret = mac;
			hmac = MACs.getHMAC(macAlgorithm);
			CIPHER_key = key;
			CIPHER_IV = IV;
			cipher = BulkCiphers.getCipher(bulkCipher, key, 0x0301);
			if (cipher is IVMode) {
				ivmode = cipher as IVMode;
				ivmode.IV = IV;
			}
		}
		
		public function decrypt(type:uint, length:uint, p:ByteArray):ByteArray {
			// decompression is a nop.
			
			if (cipherType == BulkCiphers.STREAM_CIPHER) {
				if (bulkCipher == BulkCiphers.NULL) {
					// no-op
				} else {
					cipher.decrypt(p);
				}
			} else {
				// block cipher
				var nextIV:ByteArray = new ByteArray;
				nextIV.writeBytes(p, p.length-CIPHER_IV.length, CIPHER_IV.length);
				
				cipher.decrypt(p);


				CIPHER_IV = nextIV;
				ivmode.IV = nextIV;
			}
			if (macAlgorithm!=MACs.NULL) {
				var data:ByteArray = new ByteArray;
				var len:uint = p.length - hmac.getHashSize();
				data.writeUnsignedInt(seq_hi);
				data.writeUnsignedInt(seq_lo);
				data.writeByte(type);
				data.writeShort(TLSSecurityParameters.PROTOCOL_VERSION);
				data.writeShort(len);
				if (len!=0) {
					data.writeBytes(p, 0, len);
				}
				var mac:ByteArray = hmac.compute(MAC_write_secret, data);
				// compare "mac" with the last X bytes of p.
				var mac_received:ByteArray = new ByteArray;
				mac_received.writeBytes(p, len, hmac.getHashSize());
				if (ArrayUtil.equals(mac, mac_received)) {
					// happy happy joy joy
				} else {
					throw new TLSError("Bad Mac Data", TLSError.bad_record_mac);
				}
				p.length = len;
				p.position = 0;
			}
			// increment seq
			seq_lo++;
			if (seq_lo==0) seq_hi++;
			return p;
		}
		public function encrypt(type:uint, p:ByteArray):ByteArray {
			var mac:ByteArray = null;
			if (macAlgorithm!=MACs.NULL) {
				var data:ByteArray = new ByteArray;
				data.writeUnsignedInt(seq_hi);
				data.writeUnsignedInt(seq_lo);
				data.writeByte(type);
				data.writeShort(TLSSecurityParameters.PROTOCOL_VERSION);
				data.writeShort(p.length);
				if (p.length!=0) {
					data.writeBytes(p, 0, p.length);
				}
				mac = hmac.compute(MAC_write_secret, data);
				p.position = p.length;
				p.writeBytes(mac);
			}
			p.position = 0;
			if (cipherType == BulkCiphers.STREAM_CIPHER) {
				// stream cipher
				if (bulkCipher == BulkCiphers.NULL) {
					// no-op
				} else {
					cipher.encrypt(p);
				}
			} else {
				// block cipher
				cipher.encrypt(p);
				// adjust IV
				var nextIV:ByteArray = new ByteArray;
				nextIV.writeBytes(p, p.length-CIPHER_IV.length, CIPHER_IV.length);
				CIPHER_IV = nextIV;
				ivmode.IV = nextIV;
			}
			// increment seq
			seq_lo++;
			if (seq_lo==0) seq_hi++;
			// compression is a nop.
			return p;
		}
		
	}
}