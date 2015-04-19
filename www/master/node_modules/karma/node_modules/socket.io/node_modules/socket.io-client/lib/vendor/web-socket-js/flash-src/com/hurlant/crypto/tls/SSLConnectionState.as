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
	import com.hurlant.crypto.hash.MAC;
	import com.hurlant.crypto.hash.IHash;
	import com.hurlant.crypto.symmetric.ICipher;
	import com.hurlant.crypto.symmetric.IVMode;
	import com.hurlant.util.Hex;
	import com.hurlant.util.ArrayUtil;
	
	public class SSLConnectionState implements IConnectionState {

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
		private var mac:MAC;
		
		// sequence number. uint64
		
		private var seq_lo:uint = 0x0;
		private var seq_hi:uint = 0x0;

		public function SSLConnectionState(
				bulkCipher:uint=0, cipherType:uint=0, macAlgorithm:uint=0,
				mac_enc:ByteArray=null, key:ByteArray=null, IV:ByteArray=null) {
			this.bulkCipher = bulkCipher;
			this.cipherType = cipherType;
			this.macAlgorithm = macAlgorithm;
			MAC_write_secret = mac_enc;
			mac = MACs.getMAC(macAlgorithm);
			
			CIPHER_key = key;
			CIPHER_IV = IV;
			cipher = BulkCiphers.getCipher(bulkCipher, key, 0x0300);
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
				p.position = 0;
				// block cipher
				if (bulkCipher == BulkCiphers.NULL) {
				
				} else {
					var nextIV:ByteArray = new ByteArray;
					nextIV.writeBytes(p, p.length-CIPHER_IV.length, CIPHER_IV.length);
					p.position = 0;
					cipher.decrypt(p);

					CIPHER_IV = nextIV;
					ivmode.IV = nextIV;
				}
			}
	
			if (macAlgorithm!=MACs.NULL) {
				// there will be CTX delay here as well, 
				// I should probably optmize the hell out of it
				var data:ByteArray = new ByteArray;
				var len:uint = p.length - mac.getHashSize();
				data.writeUnsignedInt(seq_hi);
				data.writeUnsignedInt(seq_lo);
				
				data.writeByte(type);
				data.writeShort(len);
				if (len!=0) {
					data.writeBytes(p, 0, len);
				}
				var mac_enc:ByteArray = mac.compute(MAC_write_secret, data);
				// compare "mac" with the last X bytes of p.
				var mac_received:ByteArray = new ByteArray;
				mac_received.writeBytes(p, len, mac.getHashSize());
				if (ArrayUtil.equals(mac_enc, mac_received)) {
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
			var mac_enc:ByteArray = null;
			if (macAlgorithm!=MACs.NULL) {
				var data:ByteArray = new ByteArray;
				// data.writeUnsignedInt(seq);
				
				// Sequence
				data.writeUnsignedInt(seq_hi);
				data.writeUnsignedInt(seq_lo);
				
				// Type
				data.writeByte(type);
				
				// Length
				data.writeShort(p.length);
				
				// The data
				if (p.length!=0) {
					data.writeBytes(p);
				}
			
				// trace("data for the MAC: " + Hex.fromArray(data));
				mac_enc = mac.compute(MAC_write_secret, data);
				// trace("MAC: " + Hex.fromArray( mac_enc ));
				p.position = p.length;
				p.writeBytes(mac_enc);
			}
			
			// trace("Record to encrypt: " + Hex.fromArray(p));
			
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
			return p;
		}
		
	}
}