/**
 * TLSPad
 * 
 * A padding implementation used by TLS
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.symmetric {
	import flash.utils.ByteArray;
	import com.hurlant.util.Hex;
	import com.hurlant.crypto.tls.TLSError;
	
	public class TLSPad implements IPad {
		private var blockSize:uint;
		
		public function TLSPad(blockSize:uint=0) {
			this.blockSize = blockSize;
		}
		public function pad(a:ByteArray):void {
			var c:uint = blockSize - (a.length+1)%blockSize;
			for (var i:uint=0;i<=c;i++) {
				a[a.length] = c;
			}
		}
		public function unpad(a:ByteArray):void {
			var c:uint = a.length%blockSize;
			if (c!=0) throw new TLSError("TLSPad::unpad: ByteArray.length isn't a multiple of the blockSize", TLSError.bad_record_mac);
			c = a[a.length-1];
			for (var i:uint=c;i>0;i--) {
				var v:uint = a[a.length-1];
				a.length--;
				if (c!=v) throw new TLSError("TLSPad:unpad: Invalid padding value. expected ["+c+"], found ["+v+"]", TLSError.bad_record_mac);
			}
			a.length--;
			// mostly ripped off from PKCS5.as, but with subtle differences
		}
		public function setBlockSize(bs:uint):void {
			blockSize = bs;
		}
	}
}